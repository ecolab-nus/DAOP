"""Microbenchmarking for CPU offloading"""

import argparse
import copy
import os
import time
import torch.nn.functional as F
import numpy as np
import torch
import subprocess
import multiprocessing
import datetime

from DAOP.ondemand_mixtral import DemandMixtral

testing_expert_num = 1


def get_cpu_temp():
    result = subprocess.run(['sensors'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')


def get_gpu_temp():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip()


def temperature_logger(log_file, interval, stop_event):
    with open(log_file, "a") as file:
        while not stop_event.is_set():
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_temp = get_cpu_temp()
            gpu_temp = get_gpu_temp()
            log_entry = f"{now}, CPU Temp: {cpu_temp}, GPU Temp: {gpu_temp}\n"
            file.write(log_entry)
            file.flush()
            time.sleep(interval)
        print("Temperature logging stopped.")


def weight_copy(model, gate_in_features, from_cpu=True):
    """Time to copy weights of an expert"""
    ret_time = []

    if from_cpu:
        expert_placeholder = copy.deepcopy(
            model.model_obj.layers[0].block_sparse_moe.experts[0]
        ).to(model._device)
        for i in range(model.n_layer):
            for j in experts_j:
                model.model_obj.layers[i].block_sparse_moe.experts[j].to("cpu")
                torch.cuda.synchronize()
                tick = time.time()
                expert_placeholder.load_state_dict(
                    model.model_obj.layers[i].block_sparse_moe.experts[j].state_dict()
                )
                torch.cuda.synchronize()
                ret_time.append(time.time() - tick)
                model.model_obj.layers[i].block_sparse_moe.experts[j].to("cpu")
    else:
        expert_placeholder = copy.deepcopy(
            model.model_obj.layers[0].block_sparse_moe.experts[0]
        ).to("cpu")
        for i in range(model.n_layer):
            for j in experts_j:
                model.model_obj.layers[i].block_sparse_moe.experts[j].to(model._device)
                torch.cuda.synchronize()
                tick = time.time()
                expert_placeholder.load_state_dict(
                    model.model_obj.layers[i].block_sparse_moe.experts[j].state_dict()
                )
                torch.cuda.synchronize()
                ret_time.append(time.time() - tick)
                model.model_obj.layers[i].block_sparse_moe.experts[j].to("cpu")
    return np.array(ret_time)


def copy_activation(model, gate_in_features, from_cpu=True):
    """Time to copy activations"""
    ret_time = []
    if from_cpu:
        for i in range(model.n_layer):
            inps = torch.randn((1, gate_in_features), dtype=model.dtype, device="cpu")
            torch.cuda.synchronize()
            tick = time.time()
            inps = inps.to(model._device)
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            del inps
    else:
        for i in range(model.n_layer):
            inps = torch.randn((1, gate_in_features), dtype=model.dtype, device=model._device)
            torch.cuda.synchronize()
            tick = time.time()
            inps = inps.to("cpu")
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            del inps
    return np.array(ret_time)


def bring_non_expert_to_gpu(model):
    """Bring non-expert layers to GPU"""
    model.lm_head.to(model._device)
    model.model_obj.embed_tokens.to(model._device)
    model.model_obj.norm.to(model._device)
    for i in range(len(model.model_obj.layers)):
        model.model_obj.layers[i].self_attn.to(model._device)
        model.model_obj.layers[i].input_layernorm.to(model._device)
        model.model_obj.layers[i].block_sparse_moe.gate.to(model._device)
        model.model_obj.layers[i].post_attention_layernorm.to(model._device)


def expert_gpu(model, gate_in_features, batch_size=1):
    """Time to execute an expert at GPU"""
    ret_time = []

    # warm up
    model.model_obj.layers[0].block_sparse_moe.experts[7].to(model._device)
    inps = torch.randn((batch_size, gate_in_features), dtype=model.dtype, device=model._device)
    weights = torch.ones((batch_size, 1), dtype=model.dtype, device=model._device)
    inps = model.model_obj.layers[0].block_sparse_moe.experts[7](inps) * weights
    model.model_obj.layers[0].block_sparse_moe.experts[7].to("cpu")
    del inps, weights
    torch.cuda.synchronize()

    for i in range(model.n_layer):
        for j in experts_j:
            model.model_obj.layers[i].block_sparse_moe.experts[j].to(model._device)
            inps = torch.randn((batch_size, gate_in_features), dtype=model.dtype, device=model._device)
            weights = torch.randn((batch_size, 1), dtype=model.dtype, device=model._device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            tick = time.time()
            inps = model.model_obj.layers[i].block_sparse_moe.experts[j](inps) * weights
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            model.model_obj.layers[i].block_sparse_moe.experts[j].to("cpu")
            del inps, weights
    return np.array(ret_time)


def expert_cpu(model, gate_in_features, batch_size=1, multithreading=False):
    """Time to execute an expert at CPU"""
    ret_time = []
    # warm up
    model.model_obj.layers[0].block_sparse_moe.experts[7].to("cpu")
    inps = torch.randn((batch_size, gate_in_features), dtype=model.dtype, device="cpu")
    weights = torch.ones((batch_size, 1), dtype=model.dtype, device="cpu")
    torch.cuda.synchronize()
    tick = time.time()
    inps = model.model_obj.run_expert_at_cpu(0, 7, inps, weights)
    del inps, weights
    torch.cuda.synchronize()

    for i in range(model.n_layer):
        for j in experts_j:
            model.model_obj.layers[i].block_sparse_moe.experts[j].to("cpu")
            inps = torch.randn((batch_size, gate_in_features), dtype=model.dtype, device="cpu")
            weights = torch.randn((batch_size, 1), dtype=model.dtype, device="cpu")
            torch.cuda.synchronize()
            tick = time.time()
            inps = model.model_obj.run_expert_at_cpu(i, j, inps, weights)
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            del inps, weights
    return np.array(ret_time)


def atten_gpu(model, attn_in_features, seq_len, batch_size=1):
    """Time to execute an atten at GPU"""
    ret_time = []

    # warm up
    model.model_obj.layers[0].self_attn.to(model._device)
    hidden_states = torch.randn((batch_size, seq_len, attn_in_features), dtype=model.dtype, device=model._device)
    hidden_states, self_attn_weights, present_key_value = model.model_obj.layers[0].self_attn(hidden_states=hidden_states,)
    model.model_obj.layers[0].self_attn.to("cpu")
    del hidden_states
    torch.cuda.synchronize()

    for i in range(model.n_layer):
        model.model_obj.layers[0].self_attn.to(model._device)    
        hidden_states = torch.randn((batch_size, seq_len, attn_in_features), dtype=model.dtype, device=model._device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        tick = time.time()
        hidden_states, self_attn_weights, present_key_value = model.model_obj.layers[0].self_attn(hidden_states=hidden_states,)
        torch.cuda.synchronize()
        ret_time.append(time.time() - tick)
        model.model_obj.layers[0].self_attn.to("cpu")
        del hidden_states
    return np.array(ret_time)


def atten_cpu(model, attn_in_features, seq_len, batch_size=1, multithreading=False):
    """Time to execute an atten at CPU"""
    ret_time = []
    # warm up
    model.model_obj.layers[0].self_attn.to("cpu")
    hidden_states = torch.randn((batch_size, seq_len, attn_in_features), dtype=model.dtype, device="cpu")
    torch.cuda.synchronize()
    tick = time.time()
    hidden_states, self_attn_weights, present_key_value = model.model_obj.layers[0].self_attn(hidden_states=hidden_states,)
    del hidden_states
    torch.cuda.synchronize()

    for i in range(model.n_layer):
        model.model_obj.layers[0].self_attn.to("cpu")
        hidden_states = torch.randn((batch_size, seq_len, attn_in_features), dtype=model.dtype, device="cpu")
        torch.cuda.synchronize()
        tick = time.time()
        hidden_states, self_attn_weights, present_key_value = model.model_obj.layers[0].self_attn(hidden_states=hidden_states,)
        torch.cuda.synchronize()
        ret_time.append(time.time() - tick)
        del hidden_states
    return np.array(ret_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    args = parser.parse_args()
    
    # multiprocessing.set_start_method('spawn')  # Required on Unix-like systems
    # # Set up logging and inference parameters
    # log_file = "temperature_log.csv"
    # interval = 3  # Log temperature every 10 seconds
    # stop_event = multiprocessing.Event()
    # temp_process = multiprocessing.Process(target=temperature_logger, args=(log_file, interval, stop_event))
    # temp_process.start()

    model = DemandMixtral(model_name_or_path=args.model, attn_implementation=None, proportion_gpu=0.2)

    gate_in_features = model.model_obj.layers[0].block_sparse_moe.gate.in_features
    experts_j = torch.randint(0, model.n_expert, (testing_expert_num,))

    attn_in_features = model.model_obj.layers[0].self_attn.q_proj.in_features 

    # print(model.model_obj.config)

    # for i in range(torch.cuda.device_count()):
    #     device = torch.device(f"cuda:{i}")
    #     torch.cuda.reset_accumulated_memory_stats(device)
    #     torch.cuda.reset_peak_memory_stats(device)
    #     torch.cuda.reset_max_memory_allocated(device)
    #     torch.cuda.reset_max_memory_cached(device)

    def format_output(array):
        return (
            f"mean: {np.mean(array) * 1000:.2f} ms, max: {max(array) * 1000:.2f} ms, min: {min(array) * 1000:.2f} ms, std: {np.std(array) * 1000:.2f} ms"
        )

    print(
        f"\n1) Weight copy, CPU -> GPU\n{format_output(weight_copy(model, gate_in_features=gate_in_features, from_cpu=True))}"
    )
    print(
        f"\n2) Weight copy, GPU -> CPU\n{format_output(weight_copy(model, gate_in_features=gate_in_features, from_cpu=False))}"
    )
    print(
        f"\n3) Activation copy, CPU -> GPU\n{format_output(copy_activation(model, gate_in_features=gate_in_features, from_cpu=True))}"
    )
    print(
        f"\n4) Activation copy, GPU -> CPU\n{format_output(copy_activation(model, gate_in_features=gate_in_features, from_cpu=False))}"
    )
    for i in [1, 2, 4, 8,]:# 16, 32, 64, 128, 256]:
        print(
            f"\n5) Expert execution, GPU batch={i}\n{format_output(expert_gpu(model, gate_in_features=gate_in_features, batch_size=i))}"
        )
    for i in [1, 2, 4, 8,]:# 16, 32, 64, 128, 256]:
        print(
            f"\n6) Expert execution, CPU batch={i}\n{format_output(expert_cpu(model, gate_in_features=gate_in_features, batch_size=i))}"
        )
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        print(
            f"\n7) Attention execution, GPU seq_len={i}\n{format_output(atten_gpu(model, attn_in_features=attn_in_features, seq_len=i, batch_size=1))}"
        )
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        print(
            f"\n8) Attention execution, CPU seq_len={i}\n{format_output(atten_cpu(model, attn_in_features=attn_in_features, seq_len=i, batch_size=1))}"
        )

    # # Signal the temperature logging process to stop
    # stop_event.set()
    # temp_process.join()
    # print("Main process has completed inference and stopped the temperature logger.")
