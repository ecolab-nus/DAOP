#!/bin/bash

# python DAOP/latency.py --model mistralai/Mixtral-8x7B-v0.1 --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 3
# python fiddler/latency.py --model mistralai/Mixtral-8x7B-v0.1 --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 3
# python DAOP/on_demand.py --model mistralai/Mixtral-8x7B-v0.1 --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 3

python DAOP/latency.py --model microsoft/Phi-3.5-MoE-instruct --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 3
python fiddler/latency.py --model microsoft/Phi-3.5-MoE-instruct --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 3
python DAOP/on_demand.py --model microsoft/Phi-3.5-MoE-instruct --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 3
