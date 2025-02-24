"""Microbenchmarking for CPU offloading"""

import argparse
import os
import torch
import time
from data import load_data_text
import logging
from DAOP_mixtral import FastMixtral


if __name__ == "__main__":

    torch.manual_seed(0)            # 为CPU设置随机种子
    torch.cuda.manual_seed_all(0)   # 为所有GPU设置随机种子

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    ) 

    parser = argparse.ArgumentParser()
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

    data_set = load_data_text(args.dataset_name, n_samples=102400)
    model = FastMixtral(args.model, args.attn_implementation, args.proportion_gpu)
    model._model.eval()
    n_sample = args.num_samples

    for input_token in [256]:
        for output_token in [256]:
            idx_text = 0
            time_count = 0
            token_count = 0
            for _ in range(n_sample):
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
                    do_sample=False,
                )
                
                torch.cuda.synchronize()
                end = time.time()
                time_count += end - start
                token_count += output_token
                prediction_text = model.tokenizer.decode(outputs_ids[0])

            print(
                f"model: {args.model}, dataset: {args.dataset_name}, attn_implementation: {args.attn_implementation}, proportion_gpu: {args.proportion_gpu}, num_samples: {args.num_samples}, "
                f"input_token: {input_token}, output_token: {output_token}, "
                f"{token_count / (time_count):.2f} token/s, "
                # f"sample output: {len(outputs_ids[0]), prediction_text}"
            )
