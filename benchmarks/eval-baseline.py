# fix numpy in colab
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.phimoe.modeling_phimoe import PhiMoESparseMoeBlock
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import argparse
import logging

sys.path.append("mixtral_offloading")
from data import load_data_text, cache_directory

model_path = {
    'mistralai/Mixtral-8x7B-v0.1': "models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841",
    'microsoft/Phi-3.5-MoE-instruct': "models--microsoft--Phi-3.5-MoE-instruct/snapshots/ae6cb90aceffd86d1e3fba55c59ec62dfc88d4a1",
}
offload_experts_a6000 = {
    'mistralai/Mixtral-8x7B-v0.1': 5,
    'microsoft/Phi-3.5-MoE-instruct': 10,
}

def main():
    # os.chdir("mixtral_offloading")
    if args.framework == 'mixtral-offloading':
        logging.info('Using mixtral-offloading')
        model = init_mixtral_offload(args.model)
    elif args.framework == 'deepspeed-mii':
        logging.info('Using deepspeed-mii')
        model = init_deepspeed_mii(args.model)
    else:
        raise ValueError(f'Unknown framework: {args.framework}')

    eval(model, args.model, args.num_samples)


def init_deepspeed_mii(model_id):
    import deepspeed
    from transformers.deepspeed import HfDeepSpeedConfig

    ds_config = {
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            }
        },
        "train_micro_batch_size_per_gpu": 1,
    }

    hfdsc = HfDeepSpeedConfig(ds_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, cache_dir=cache_directory, trust_remote_code=True,)

    if 'Phi' in model_id:
        deepspeed.utils.set_z3_leaf_modules(model, [PhiMoESparseMoeBlock])
    elif 'Mixtral' in model_id:
        deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    model.eval()

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    return model


def init_mixtral_offload(model_name):
    from hqq.core.quantize import BaseQuantizeConfig
    from mixtral_offloading.src.build_model import OffloadConfig, QuantConfig, build_model

    quantized = False
    
    if not quantized:
        state_path = cache_directory + model_path[model_name]
        model_name = model_name
    # else:
    #     state_path = "../Mixtral-8x7B-v0.1-offloading-demo"
    #     model_name = "lavawolfiee/Mixtral-8x7B-v0.1-offloading-demo"

    config = AutoConfig.from_pretrained(model_name)
    # logging.info(f'init_mixtral_offload__config {config}')

    device = torch.device("cuda:0")

    ##### Change this to 5 if you have only 12 GB of GPU VRAM #####
    offload_per_layer = offload_experts_a6000[model_name]
    ###############################################################

    num_experts = config.num_local_experts

    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )

    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256

    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )

    if quantized:
        quant_config = QuantConfig(
            ffn_config=ffn_config,
            attn_config=attn_config)
    else:
        quant_config = None

    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )
    return model


def eval(model, model_name, num_samples):
    import random
    import json
    import time

    device = torch.device("cuda:0")
    texts = load_data_text('sharegpt', n_samples=4096)
    logging.info(f'n of input {len(texts)}')
    n_sample = num_samples

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for input_token in [256]:
        for output_token in [256]:
            idx_text = 0
            time_sum = 0
            num_tokens = 0
            # logging.info(
            #     f'evaluating -- input_token: {input_token}, output_token: {output_token}')
            for _ in range(n_sample):
                while True:
                    text = texts[idx_text]["prompt"]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
                
                torch.cuda.synchronize()
                start_time = time.time()

                result = model.generate(
                    input_ids=input_ids[:, :input_token],
                    max_new_tokens=output_token,
                    min_new_tokens=output_token,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )

                torch.cuda.synchronize()
                end_time = time.time()
                time_sum += end_time - start_time
                # count the number of tokens in the output
                num_tokens += result["sequences"].shape[1]
                del input_ids, result

            logging.info(
                # f'*******************\n'
                f'input_token: {input_token}, output_token: {output_token}, '
                f'time: {time_sum / n_sample:.2f} seconds for {n_sample} sequences, '
                f'token/s: {output_token / (time_sum / n_sample):.2f}, '
                # f'sample output: {tokenizer.decode(result["sequences"][0])}'
                # f'*******************\n'
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--quantized', type=bool, default=False,
        help='Whether to use quantized model in mixtral-offloading.'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default='mixtral-offloading',
        choices=[
            'mixtral-offloading',
            'deepspeed-mii'],
        help='Which framework to use for evaluation.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mixtral-8x7B-v0.1',
        choices=[
            'mistralai/Mixtral-8x7B-v0.1',
            'microsoft/Phi-3.5-MoE-instruct'],
        help='Which model to use for evaluation.'
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10,
    )
    args = parser.parse_args()

    # save log to file
    logging.basicConfig(level=logging.INFO, filename='generated/mixtral_offloading_deepspeed_mii.log',)
    main()
