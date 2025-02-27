#!/bin/bash


# ################################# Platform-aware Data Collection on NVIDIA A6000 CPU #################################
# taskset -c 0-17 python DAOP/measure/microbench.py --model mistralai/Mixtral-8x7B-v0.1
# taskset -c 0-17 python DAOP/measure/microbench.py --model microsoft/Phi-3.5-MoE-instruct

sudo swapoff -a
sudo swapon /swapfile
sudo swapon --show
free -h

################################# To limit the program to only utilize 18 logical CPU cores (9 physical CPU cores) within the same socket (NVIDIA A6000 CPU) #################################
# Using Only One Logical CPU per Physical Core
# Avoid Resource Contention: Since only one thread is running on each physical core, there is no contention for shared resources.
# Improved Performance for Heavy Workloads: If your program is CPU-bound and performs heavy computations, this approach ensures that the full core resources are dedicated to a single thread.
# More Predictable Performance: Ideal for real-time or latency-sensitive applications.

################################# On NVIDIA A6000 server, NUMA node 0 has 36 logical cores, with 0-17 and 18-35 being pairs from hyperthreading #################################
# Set Soft Affinity with Multiple Logical Cores: Bind Only One Logical Core per Physical Core:
# numactl --physcpubind=0-17 --localalloc python my_program.py
# --physcpubind=0-17 binds to the first logical core of each physical core. Avoids resource contention by not using the second hyperthread. The process prefers to run on these cores, but the scheduler can move it if needed.
# --localalloc: Ensures memory allocations are local to the NUMA node of the bound CPU cores. Minimizes memory latency by keeping memory accesses local.

# numactl --physcpubind=0-17 --localalloc python DAOP/daop_infer.py --model mistralai/Mixtral-8x7B-v0.1 --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 30
# numactl --physcpubind=0-17 --localalloc python fiddler/fiddler_infer.py --model mistralai/Mixtral-8x7B-v0.1 --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 30
# numactl --physcpubind=0-17 --localalloc python DAOP/on_demand.py --model mistralai/Mixtral-8x7B-v0.1 --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 30

numactl --physcpubind=0-17 --localalloc python DAOP/daop_infer.py --model microsoft/Phi-3.5-MoE-instruct --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 30
numactl --physcpubind=0-17 --localalloc python fiddler/fiddler_infer.py --model microsoft/Phi-3.5-MoE-instruct --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 30
# numactl --physcpubind=0-17 --localalloc python DAOP/on_demand.py --model microsoft/Phi-3.5-MoE-instruct --dataset_name sharegpt --proportion_gpu 0.99 --num_samples 30
