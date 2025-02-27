import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import popular_experts_small, popular_experts_phi, popular_experts_big, designated_cache_size, cache_directory
from model import Daop_MixtralModel, Daop_PhiMoEModel

np.random.seed(0)
torch.manual_seed(0)            # 为CPU设置随机种子
torch.cuda.manual_seed_all(0)   # 为所有GPU设置随机种子


class DaopMoE:
    def __init__(self, model_name_or_path: str, attn_implementation: str, proportion_gpu: int = 0.95):
        self.dtype = torch.bfloat16
        self._device = torch.device("cuda:0")
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            attn_implementation=attn_implementation,
            # quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
            use_cache=True,
            cache_dir=cache_directory,
            trust_remote_code=True,
        )
        config = self._model.config

        self.lm_head = self._model.lm_head
        # self._model.generation_config.pad_token_id = self._model.generation_config.eos_token_id
        
        if 'Phi' in model_name_or_path:
            self._model.model = Daop_PhiMoEModel(self._model.model, config, self._device)
            print('Fast_PhiMoE Replacement Done!')
            self.popular_experts = popular_experts_phi
        elif 'Mixtral' in model_name_or_path:
            self._model.model = Daop_MixtralModel(self._model.model, config, self._device)
            print('Fast_Mixtral Replacement Done!')
            if '8x7B' in model_name_or_path:
                self.popular_experts = popular_experts_small
            elif '8x22B' in model_name_or_path:
                self.popular_experts = popular_experts_big

        self.model_obj = self._model.model
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        
        if self.tokenizer.pad_token:
            pass
        elif self.tokenizer.unk_token:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.tokenizer.eos_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.n_layer = len(self.model_obj.layers)
        self.n_expert = len(self.model_obj.layers[0].block_sparse_moe.experts)

        self.bring_non_expert_to_gpu()

        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.proportion_gpu = proportion_gpu
        n_expert_on_gpu = min(self.calc_n_expert_on_gpu(), designated_cache_size)
        print(f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}")

        self.set_expert_loc(n_expert_on_gpu, self.popular_experts)
        print(self.expert_loc)
        self.bring_expert_to_gpu()
        print("Model is ready.")


    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self._device)
        self.model_obj.embed_tokens.to(self._device)
        self.model_obj.norm.to(self._device)
        for i in range(len(self.model_obj.layers)):
            self.model_obj.layers[i].self_attn.to(self._device)
            self.model_obj.layers[i].input_layernorm.to(self._device)
            self.model_obj.layers[i].block_sparse_moe.gate.to(self._device)
            self.model_obj.layers[i].post_attention_layernorm.to(self._device)
            # only model.layers[i].block_sparse_moe.experts is on CPU

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """Set the location of experts"""
        if popular_experts is None:
            # list of (i_layer, i_expert) in the order of popularity determined based on profile
            popular_experts = []

        sign_experts = {}
        supplement_cache_num = 0
        layer_cache = (n_expert_on_gpu - supplement_cache_num) // self.n_layer
        # Distribute popular experts to layers until the layer's cache limit is reached
        if layer_cache >= 1:
            for layer_index in range(self.n_layer):
                cache_count = 0
                for i, (i_layer, i_expert) in enumerate(popular_experts):
                    if i_layer == layer_index and i not in sign_experts:
                        if cache_count >= layer_cache:
                            break
                        else:
                            self.expert_loc[i_layer, i_expert] = 1
                            sign_experts[i] = True
                            cache_count += 1
                            supplement_cache_num += 1
        
        # Handle any remaining experts if they have not been placed and there is still capacity
        cur_experts = supplement_cache_num
        for i, (i_layer, i_expert) in enumerate(popular_experts):
            if cur_experts < n_expert_on_gpu and i not in sign_experts:
                self.expert_loc[i_layer, i_expert] = 1
                sign_experts[i] = True
                cur_experts += 1


    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model_obj.layers[i].block_sparse_moe.experts[j].to(self._device)


    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1


    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self.model_obj.layers[0].block_sparse_moe.experts[0].parameters()
        )
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self._device).total_memory
        free_mem = total_mem * self.proportion_gpu - torch.cuda.memory_allocated(self._device) #0.95
        return int((free_mem) // (n_param * 2))
