"""Microbenchmarking for CPU offloading"""
import argparse
import os
# import random
import torch

from data import load_data_text

from fiddler.fiddler_mixtral import FiddlerMixtral
import logging
import time

if __name__ == "__main__":

    torch.manual_seed(0)            # 为CPU设置随机种子
    torch.cuda.manual_seed_all(0)   # 为所有GPU设置随机种子

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    ) #, filename='prediction.log'

    parser = argparse.ArgumentParser()
    # os.chdir("mixtral_offloading")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sharegpt",
        help="Dataset Name",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default=None,
        choices=[None, "eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument(
        "--cpu_offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: GPU-only (baseline), 1: fiddler.",
    )
    parser.add_argument(
        "--proportion_gpu",
        type=float, 
        default=0.95,
        help="porpotion of device memory used for execution",
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10,
    )

    args = parser.parse_args()
    # path_json = home_directory + "/MoE_Infer/data/ShareGPT_V3_unfiltered_cleaned_split.json"
    # with open(path_json, "r") as f:
    #     data = json.load(f)
    # texts = []
    # for d in data:
    #     if len(d["conversations"]) == 0:
    #         continue
    #     # the input of the first round
    #     texts.append(" ".join(d["conversations"][0]["value"].split()))
    # random.seed(0)
    # random.shuffle(texts)

    # path_json = home_directory + "/MoE_Infer/data/ShareGPT_V3_filtered_shuffled.json"
    # n_samples = 1024
    # with open(path_json, "r") as f:
    #     texts = json.load(f)
    # texts = random.sample(texts, k=min(n_samples, len(texts)))

    data_set = load_data_text(args.dataset_name, n_samples=102400)
    model = FiddlerMixtral(args.model, args.attn_implementation, args.cpu_offload, args.proportion_gpu)
    model._model.eval()
    n_sample = args.num_samples

    for input_token in [256]:#[64, 128, 256]: [1024, 2048, 4096, 8192, 16384]
        for output_token in [256]:#[128, 256, 512]: [64, 256, 1024, 4096, 16384]
            # if output_token > 2 * input_token:
            #     continue
            idx_text = 0
            # prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            time_count = 0
            token_count = 0
            for sample_i in range(n_sample):
                while True:
                    text = data_set[idx_text]["prompt"]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                
                input_ids = model.tokenizer.encode(text, return_tensors='pt').to(model._device)
                
                torch.cuda.synchronize()
                start = time.time()

                outputs_ids = model._model.generate(
                    input_ids=input_ids[:, :input_token],
                    max_new_tokens=output_token,
                    min_new_tokens=output_token,
                    pad_token_id=model.tokenizer.pad_token_id, 
                    use_cache=True,
                    # do_sample=False,
                    ############
                    # do_sample=True,
                    # temperature=0.9,
                    # top_p=0.9,
                    #########
                    num_beams=5,  # Number of beams
                    early_stopping=True,  # Stops early if all beams produce EOS
                )
                
                torch.cuda.synchronize()
                end = time.time()
                time_count += end - start
                token_count += output_token
                prediction_text = model.tokenizer.decode(outputs_ids[0])
                # print('Output:', prediction_text)
                # print(f"{idx_text, len(model.layer_noe_count[0]), len(model.layer_noe_count[1]), len(model.layer_noe_count[2])}")
                # for option_i in range(3):
                #     print(f"{option_i, model.layer_type_count[option_i]}, layer_noe: {np.average(model.layer_noe_count[option_i])}, layer_e: {np.average(model.layer_e_count[option_i])}")
                #     # input()

                # print((f"{sample_i}-{input_token}-{output_token}-"
                #        f"{model._model.model.kv_cache['prefill']}-"
                #        f"{'-'.join(model._model.model.kv_cache[term_num] for term_num in model._model.model.term_num_list)}"))
                # re-initialization
                model._model.model.initialize_info()

            print(
                f"model: {args.model}, dataset: {args.dataset_name}, attn_implementation: {args.attn_implementation}, cpu_offload: {args.cpu_offload}, proportion_gpu: {args.proportion_gpu}, num_samples: {args.num_samples}, "
                f"input_token: {input_token}, output_token: {output_token}, "
                f"{token_count / (time_count):.2f} token/s, "
                # f"sample output: {len(outputs_ids[0]), prediction_text}"
            )
