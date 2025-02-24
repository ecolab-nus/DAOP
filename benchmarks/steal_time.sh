#!/bin/bash
# python benchmarks/eval-baseline.py --framework=deepspeed-mii --num_samples 30
# python benchmarks/eval-baseline.py --framework=mixtral-offloading --num_samples 30

python benchmarks/eval-baseline.py --framework=deepspeed-mii --model microsoft/Phi-3.5-MoE-instruct --num_samples 3
python benchmarks/eval-baseline.py --framework=mixtral-offloading --model microsoft/Phi-3.5-MoE-instruct --num_samples 3