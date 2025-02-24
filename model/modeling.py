# import warnings
from typing import Optional, Tuple, Union, List
# import time
import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
import copy

from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
    MixtralPreTrainedModel,)

from transformers.models.phimoe.modeling_phimoe import (
    PhiMoEConfig,
    MoeModelOutputWithPast,
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    PhiMoEPreTrainedModel,
    sparsemixer,)

from queue import Queue
from transformers.cache_utils import Cache, DynamicCache
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
from data import latency_cpu, latency_gpu, swap_in_out, largest_num, first_predict_layer
import heapq

class Fast_MixtralModel(MixtralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MixtralDecoderLayer`]

    Args:
        config: MixtralConfig
    """

    def __init__(self, model, config: MixtralConfig, _device):
        super().__init__(config)

        # self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # self.layers = nn.ModuleList(
        #     [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        # )
        # self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.gradient_checkpointing = False
        # # Initialize weights and apply final processing
        # self.post_init()
    
        self.padding_idx = model.padding_idx
        self.vocab_size = model.vocab_size
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self._use_flash_attention_2 = None #model._use_flash_attention_2
        self.norm = model.norm
        self.gradient_checkpointing = model.gradient_checkpointing

        self.n_layer = config.num_hidden_layers
        self.n_expert = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # TODO: find this value based on device config
        self.executor = ThreadPoolExecutor(2)
        self.dev = _device

        self.expert_placeholder = copy.deepcopy(
            self.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        
        self.activate_matrix = np.zeros((self.n_layer, self.n_expert), dtype=int)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._use_flash_attention_2 and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        hidden_states, all_hidden_states, all_self_attns, all_router_logits, next_decoder_cache = self.mixtral_forward(hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            use_cache=use_cache,)
        
        # for decoder_layer in self.layers:
        #     if output_hidden_states:
        #         all_hidden_states += (hidden_states,)

        #     if self.gradient_checkpointing and self.training:
        #         layer_outputs = self._gradient_checkpointing_func(
        #             decoder_layer.__call__,
        #             hidden_states,
        #             attention_mask,
        #             position_ids,
        #             past_key_values,
        #             output_attentions,
        #             output_router_logits,
        #             use_cache,
        #         )
        #     else:
        #         layer_outputs = decoder_layer(
        #             hidden_states,
        #             attention_mask=attention_mask,
        #             position_ids=position_ids,
        #             past_key_value=past_key_values,
        #             output_attentions=output_attentions,
        #             output_router_logits=output_router_logits,
        #             use_cache=use_cache,
        #         )

        #     hidden_states = layer_outputs[0]

        #     if use_cache:
        #         next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        #     if output_attentions:
        #         all_self_attns += (layer_outputs[1],)

        #     if output_router_logits:
        #         all_router_logits += (layer_outputs[-1],)
                

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
    
    @torch.no_grad()
    def mixtral_forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        ########## added for expert pre-calculation ##########
        pre_calculated = {i:False for i in range(self.n_layer)}
        future_work_list = Queue()

        ### annotation
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # if sequence_length == 1:
        #     print('decode')
        # else:
        #     print('prefill')

        for i_layer, decoder_layer in enumerate(self.layers):

            ########## added for expert pre-calculation ##########
            i_predict_layer = min(i_layer+1, self.n_layer-1)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                )
            else:
                # layer_outputs = decoder_layer(
                #     hidden_states,
                #     attention_mask=attention_mask,
                #     position_ids=position_ids,
                #     past_key_value=past_key_value,
                #     output_attentions=output_attentions,
                #     output_router_logits=output_router_logits,
                #     use_cache=use_cache,
                # )
                
                residual = hidden_states

                hidden_states = decoder_layer.input_layernorm(hidden_states)

                # Self Attention
                hidden_states, self_attn_weights, present_key_value = decoder_layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states
                hidden_states = decoder_layer.post_attention_layernorm(hidden_states)

                # hidden_states, router_logits = decoder_layer.block_sparse_moe(hidden_states)

                batch_size, sequence_length, hidden_dim = hidden_states.shape
                hidden_states = hidden_states.view(-1, hidden_dim)
                is_decode = True if sequence_length == 1 else False
                experts = decoder_layer.block_sparse_moe.experts

                # # intermediate variable to store the output of experts
                final_hidden_states = torch.zeros(
                    (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=self.dev
                )
                final_hidden_states_cpu = torch.zeros(
                    (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=self.dev
                )

                if not is_decode:
                    router_logits = decoder_layer.block_sparse_moe.gate(hidden_states)
                    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                    # we cast back to the input dtype
                    routing_weights = routing_weights.to(hidden_states.dtype)
                else:
                    if i_layer < first_predict_layer: # 0, 1, 2, 3, 4
                        router_logits, experts_cpu, experts_gpu, routing_weights_cpu, routing_weights_gpu = self.assign_gate_on_inps(i_layer, hidden_states, False)
                    else: 
                        experts_gpu = predict_experts_gpu
                        experts_cpu, routing_weights_cpu = [], []
                        routing_weights_gpu = routing_weights_dict_gpu
                        router_logits = predict_router_logits

                    if i_layer >= first_predict_layer - 1 and i_layer < self.n_layer - 1: # 4-30
                        ####################### expert prediction using next layer's gate function #######################
                        predict_router_logits, predict_experts_cpu, predict_experts_gpu, routing_weights_dict_cpu, routing_weights_dict_gpu = self.assign_gate_on_inps(i_predict_layer, hidden_states, True)

                    ####### expert pre-calculation #######
                    if i_layer >= first_predict_layer - 1 and i_layer < self.n_layer - 1 and len(predict_experts_cpu) > 0:  # 4-30
                        pre_calculated[i_predict_layer] = True
                        future_work_list.put(self.executor.submit(self.pre_run_expert_in_thread_cpu, hidden_states, i_predict_layer, predict_experts_cpu, routing_weights_dict_cpu))      
                    else: 
                        pre_calculated[i_predict_layer] = False


                # dominated experts on GPU, expert pre-calculate on cpu
                if not is_decode:

                    expert_mask = torch.nn.functional.one_hot(
                        selected_experts, num_classes=self.n_expert
                    ).permute(2, 1, 0)              

                    # first, calculate the number of tokens for each expert
                    idxs, top_2s = [], []
                    cost_per_expert = np.zeros((self.n_expert, 2), dtype=float)  # 0: CPU, 1: GPU
                           
                    hot_indices = []
                    is_gpu = torch.tensor([1 if next(experts[i_expert].parameters()).device == self.dev else sequence_length for i_expert in range(self.n_expert)])
                    is_cpu = torch.tensor([0 if next(experts[i_expert].parameters()).device == self.dev else 1 for i_expert in range(self.n_expert)])
                    

                    for i_expert in range(self.n_expert):
                        idx, top_2 = torch.where(expert_mask[i_expert])
                        idxs.append(idx)
                        top_2s.append(top_2)  # the token indices to be processed on this expert
                        # expected latency at CPU: number of token * cost_at_cpu
                        # expected latency at GPU: cost_at_gpu (constant)
                        cost_per_expert[i_expert, 0] = top_2.shape[0] * latency_cpu
                        cost_per_expert[i_expert, 1] = latency_gpu
                        if next(experts[i_expert].parameters()).device == self.dev:
                            # if the expert is in GPU, the latency at GPU is approximately 0
                            cost_per_expert[i_expert, 1] = 0

                        hot_indices.append(top_2.shape[0])
                    

                    # hot_expert = int(np.argmax(np.multiply(hot_indices, is_cpu)))
                    # cold_expert = int(np.argmin(np.multiply(hot_indices, is_gpu)))
                    hot_experts = heapq.nlargest(largest_num, range(self.n_expert), np.multiply(hot_indices, is_cpu.tolist()).take)
                    cold_experts = heapq.nsmallest(largest_num, range(self.n_expert), np.multiply(hot_indices, is_gpu.tolist()).take)
                    # print('\n', i_layer, 'hot_indices:', hot_indices, 'is_gpu:', is_gpu)
                    # print(hot_experts, cold_experts)

                    for hot_expert, cold_expert in zip(hot_experts, cold_experts):
                        if hot_indices[hot_expert] > swap_in_out * hot_indices[cold_expert] and is_cpu[hot_expert] and is_gpu[cold_expert]==1:
                            assert not next(experts[hot_expert].parameters()).device == self.dev and next(experts[cold_expert].parameters()).device == self.dev

                            # print(f'{i_layer}, swap {hot_expert, hot_indices[hot_expert]} to GPU; swap {cold_expert, hot_indices[cold_expert]} out')

                            experts[cold_expert].to("cpu", non_blocking=True)
                            experts[hot_expert].to(self.dev, non_blocking=True)
                            cost_per_expert[cold_expert, 1] = latency_gpu
                            cost_per_expert[hot_expert, 1] = 0

                            assert next(experts[hot_expert].parameters()).device == self.dev and not next(experts[cold_expert].parameters()).device == self.dev
                    
                    # second, partition experts processing between CPU and GPU so that we can minimize:
                    # max(sum of cost at CPU, sum of cost at GPU)
                    # greedy algorithm is just as there are only 8 experts for Mixtral
                    best_config = -1
                    best_cost = float("inf")
                    for config in range(1 << self.n_expert):
                        sum_cost = 0
                        for i_expert in range(self.n_expert):
                            if (config >> i_expert) & 1:
                                sum_cost += cost_per_expert[i_expert, 0]
                            else:
                                sum_cost += cost_per_expert[i_expert, 1]
                        
                        if sum_cost < best_cost:
                            best_cost = sum_cost
                            best_config = config

                    # then, we can offload the experts according to the best configuration
                    cpu_experts = []
                    gpu_experts = []
                    for i_expert in range(self.n_expert):
                        if (best_config >> i_expert) & 1:
                            cpu_experts.append(i_expert)
                        else:
                            gpu_experts.append(i_expert)

                    def run_expert_in_thread():
                        for i_expert in cpu_experts:
                            top_2_list = top_2s[i_expert].tolist()
                            idx_list = idxs[i_expert].tolist()
                            current_state = hidden_states[None, top_2_list].reshape(-1, hidden_dim)
                            current_state = self.run_expert_at_cpu(
                                i_layer,
                                i_expert,
                                current_state.to("cpu", non_blocking=True),
                                routing_weights[top_2_list, idx_list, None].to("cpu", non_blocking=True),
                            )
                            final_hidden_states_cpu.index_add_(
                                0,
                                top_2s[i_expert].to(self.dev, non_blocking=True),
                                current_state.to(self.dev, non_blocking=True),
                            )

                    if len(cpu_experts) > 0:
                        work_obj = [self.executor.submit(run_expert_in_thread)]

                    for i_expert in gpu_experts:
                        top_2_list = top_2s[i_expert].tolist()
                        idx_list = idxs[i_expert].tolist()
                        current_state = hidden_states[None, top_2_list].reshape(-1, hidden_dim)
                        if next(experts[i_expert].parameters()).device == self.dev:
                            # current_state = experts[i_expert](
                            #     current_state, routing_weights[top_2_list, idx_list, None]
                            # )
                            current_state = experts[i_expert](current_state) * routing_weights[top_2_list, idx_list, None]
                            
                        else:
                            self.expert_placeholder.load_state_dict(
                                experts[i_expert].state_dict()
                            )
                            # current_state = self.expert_placeholder(
                            #     current_state, routing_weights[top_2_list, idx_list, None]
                            # )
                            current_state = self.expert_placeholder(current_state) * routing_weights[top_2_list, idx_list, None]
                            
                            
                            # cold_expert = int(np.argmin(np.multiply(hot_indices, is_gpu)))
                            # is_gpu[cold_expert] = 1000
                            # is_gpu[i_expert] = 1
                            # experts[cold_expert].to("cpu", non_blocking=True)
                            # experts[i_expert].to(self.dev, non_blocking=True)
                            # current_state = experts[i_expert](
                            #     current_state, routing_weights[top_2_list, idx_list, None]
                            # )

                            # print('is_gpu:', is_gpu)
                            # print('swap', i_expert, 'to GPU; swap', cold_expert, 'out')
                            # print('i_expert:', i_expert, 'tokens:', hot_indices[i_expert], 'cold_expert:', cold_expert, 'tokens:', hot_indices[cold_expert])
                            # torch.cuda.synchronize()
                            # print('verification:', next(experts[i_expert].parameters()).device == self.dev, not next(experts[cold_expert].parameters()).device == self.dev)


                        final_hidden_states.index_add_(
                            0,
                            top_2s[i_expert].to(self.dev, non_blocking=True),
                            current_state.to(self.dev, non_blocking=True),
                        )
                        
                    if len(cpu_experts) > 0:  # gpu+cpu, cpu
                        wait(fs=work_obj, timeout=None, return_when=ALL_COMPLETED)
                        final_hidden_states += final_hidden_states_cpu
                        
                else:
                    # ####### expert pre-calculation #######
                    # if i_layer > 0 and i_layer < self.n_layer - 1 and len(predict_experts_cpu) > 0:  # 1-30
                    #     future_work_list.put(self.executor.submit(self.pre_run_expert_in_thread_cpu, inps, i_predict_layer, predict_experts_cpu, routing_weights_dict_cpu))  
                    #     pre_calculated[i_predict_layer] = True
                    # else: 
                    #     pre_calculated[i_predict_layer] = False

                    ####### decode stage with offloading -- current layer expert execution #######
                    for expert_i in experts_cpu:
                        final_hidden_states += self.run_expert_at_cpu(
                            i_layer,
                            expert_i,
                            hidden_states.to("cpu", non_blocking=True),
                            routing_weights_cpu[expert_i].to("cpu", non_blocking=True),
                        ).to(self.dev, non_blocking=True)
                    
                    for expert_i in experts_gpu:
                        # final_hidden_states += experts[expert_i](hidden_states, routing_weights_gpu[expert_i])
                        final_hidden_states += experts[expert_i](hidden_states) * routing_weights_gpu[expert_i]

                    ####### Receive expert pre-calculation results from CPU #######
                    if i_layer >= first_predict_layer and pre_calculated[i_layer]: # 4-30
                        done, not_done = wait(fs=[future_work_list.get()], timeout=None, return_when=FIRST_COMPLETED)
                        final_hidden_states_cpu = done.pop().result()
                        final_hidden_states += final_hidden_states_cpu.to(self.dev, non_blocking=False)
                        pre_calculated[i_layer] = False

                hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

                hidden_states = residual + hidden_states
                layer_outputs = (hidden_states,)

                if output_attentions:
                    layer_outputs += (self_attn_weights,)

                if use_cache:
                    layer_outputs += (present_key_value,)

                if output_router_logits:
                    layer_outputs += (router_logits,)

                # addition because there's residual connection over moe layer
                # end of one layer


            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

    
        return hidden_states, all_hidden_states, all_self_attns, all_router_logits, next_decoder_cache


    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        # return self.layers[i_layer].block_sparse_moe.experts[i_expert](
        #     inps, routing_weights
        # )
        return self.layers[i_layer].block_sparse_moe.experts[i_expert](inps) * routing_weights
        
    

    def assign_gate_on_inps(self, i_predict_layer, hidden_states, pre_calulation):
        predict_layer_experts = self.layers[i_predict_layer].block_sparse_moe.experts # 1 - 31
        predict_router_logits = self.layers[i_predict_layer].block_sparse_moe.gate(hidden_states)
        predict_routing_weights = F.softmax(predict_router_logits, dim=1)
        top_weights, top_experts = torch.topk(predict_routing_weights, 2, dim=-1, sorted=True)
        top_weights /= top_weights.sum(dim=-1, keepdim=True)
        # print('before:', top_experts, top_weights) #tensor([[2, 4]], device='cuda') tensor([[0.6992, 0.2988]], device='cuda', dtype=torch.bfloat16)

        routing_weights_dict_cpu, routing_weights_dict_gpu = {}, {}
        predict_experts_cpu, predict_experts_gpu = [], []
        for index in range(len(top_experts[0])):
            i_expert = int(top_experts[0][index])
            if next(predict_layer_experts[i_expert].parameters()).device == self.dev:
                predict_experts_gpu.append(i_expert)
                routing_weights_dict_gpu[i_expert] = top_weights[:, index, None]
                ### added by yujie
                self.activate_matrix[i_predict_layer][i_expert] += 1
            else:
                predict_experts_cpu.append(i_expert)
                routing_weights_dict_cpu[i_expert] = top_weights[:, index, None]
                ### added by yujie
                self.activate_matrix[i_predict_layer][i_expert] += 1

        # if two experts are all on cpu, then select one most similar expert on gpu as an alternative option
        if len(predict_experts_cpu) == 2 and pre_calulation:
            is_gpu = torch.tensor([next(predict_layer_experts[i_expert].parameters()).device == self.dev for i_expert in range(self.n_expert)])
            # print(i_predict_layer, 'is_gpu:', is_gpu, predict_routing_weights, predict_experts_cpu)
            if is_gpu.any():
                
                # hot_expert = predict_experts_cpu[0]

                # # last_weights_cpu, last_indices_gpu = torch.topk(predict_routing_weights[0][is_gpu].flip(dims=[0]), 1, dim=-1)
                # # global_indices = torch.arange(self.n_expert).to(self.dev, non_blocking=True)
                # # global_indices_gpu = global_indices[is_gpu][last_indices_gpu]
                # # cold_expert = global_indices_gpu[0].item()

                # last_indices_gpu = np.argmin(self.activate_matrix[i_predict_layer][is_gpu])
                # global_indices = torch.arange(self.n_expert).to(self.dev, non_blocking=True)
                # global_indices_gpu = global_indices[is_gpu][torch.tensor((last_indices_gpu))]
                # cold_expert = global_indices_gpu.item()
                
                # # print(i_predict_layer, 'is_gpu:', is_gpu, predict_routing_weights, predict_experts_cpu, self.activate_matrix[i_predict_layer][is_gpu], last_indices_gpu, global_indices_gpu)

                popped_expert = predict_experts_cpu.pop()

                # if self.activate_matrix[i_predict_layer][hot_expert] >  self.activate_matrix[i_predict_layer][cold_expert] and self.activate_matrix[i_predict_layer][cold_expert] / sum(self.activate_matrix[i_predict_layer]) < swap_in_out - 1:
                #     assert not next(predict_layer_experts[hot_expert].parameters()).device == self.dev and next(predict_layer_experts[cold_expert].parameters()).device == self.dev     
                    
                #     predict_layer_experts[cold_expert].to("cpu", non_blocking=True)
                #     predict_layer_experts[hot_expert].to(self.dev, non_blocking=True)
                #     predict_experts_cpu[0] = popped_expert
                #     predict_experts_gpu.append(hot_expert)
                #     routing_weights_dict_gpu[hot_expert] = routing_weights_dict_cpu[hot_expert]
                #     routing_weights_dict_cpu.pop(hot_expert, None)

                #     assert next(predict_layer_experts[hot_expert].parameters()).device == self.dev and not next(predict_layer_experts[cold_expert].parameters()).device == self.dev
                #     # print(f'{i_predict_layer}, swap {hot_expert, predict_routing_weights[0][hot_expert]} to GPU; swap {cold_expert, predict_routing_weights[0][cold_expert]} out')
                # else:
                top_weights_gpu, top_indices_gpu = torch.topk(predict_routing_weights[0][is_gpu], 1, dim=-1)
                global_indices = torch.arange(self.n_expert).to(self.dev, non_blocking=True)
                global_indices_gpu = global_indices[is_gpu][top_indices_gpu]
                predict_expert_index_gpu = global_indices_gpu[0].item()
                # print(predict_routing_weights, predict_experts_cpu, is_gpu, top_weights_gpu)
                
                # if predict_routing_weights[0][predict_experts_cpu[1]] - min(top_weights_gpu) < 0.05:
                #     # Find the most similar GPU expert if both are present
                #     routing_weights_dict_cpu.pop(predict_experts_cpu.pop(), None)
                #     predict_expert_index_cpu = predict_experts_cpu[0]
                #     predict_experts_gpu.append(predict_expert_index_gpu)

                #     top_weights_dict = torch.tensor([[predict_routing_weights[0][predict_expert_index_gpu], predict_routing_weights[0][predict_expert_index_cpu]]]).to(self.dev, non_blocking=True)
                #     top_weights_dict /= top_weights_dict.sum(dim=-1, keepdim=True)
                #     routing_weights_dict_gpu[predict_expert_index_gpu] = top_weights_dict[:, 0, None]
                #     routing_weights_dict_cpu[predict_expert_index_cpu] = top_weights_dict[:, 1, None]

                # Find the larger CPU expert if both are present
                # popped_expert = predict_experts_cpu.pop()
                routing_weights_dict_cpu.pop(popped_expert, None)
                predict_expert_index_cpu = predict_experts_cpu[0]
                predict_experts_gpu.append(predict_expert_index_gpu)
                ### added by yujie
                self.activate_matrix[i_predict_layer][predict_expert_index_gpu] += 1
                self.activate_matrix[i_predict_layer][popped_expert] -= 1

                top_weights_dict = torch.tensor([[predict_routing_weights[0][predict_expert_index_gpu], predict_routing_weights[0][predict_expert_index_cpu]]]).to(self.dev, non_blocking=True)
                top_weights_dict /= top_weights_dict.sum(dim=-1, keepdim=True)
                routing_weights_dict_gpu[predict_expert_index_gpu] = top_weights_dict[:, 0, None]
                routing_weights_dict_cpu[predict_expert_index_cpu] = top_weights_dict[:, 1, None]

                # print(predict_experts_cpu, predict_experts_gpu, routing_weights_dict_cpu, routing_weights_dict_gpu)
                # input()
        return predict_router_logits, predict_experts_cpu, predict_experts_gpu, routing_weights_dict_cpu, routing_weights_dict_gpu
    
        # hot_indices = []
        # is_gpu = torch.tensor([1 if next(experts[i_expert].parameters()).device == self.dev else sequence_length for i_expert in range(self.n_expert)])
        # is_cpu = torch.tensor([0 if next(experts[i_expert].parameters()).device == self.dev else 1 for i_expert in range(self.n_expert)])
        
        # # hot_expert = int(np.argmax(np.multiply(hot_indices, is_cpu)))
        # # cold_expert = int(np.argmin(np.multiply(hot_indices, is_gpu)))
        # hot_experts = heapq.nlargest(largest_num, range(self.n_expert), np.multiply(hot_indices, is_cpu.tolist()).take)
        # cold_experts = heapq.nsmallest(largest_num, range(self.n_expert), np.multiply(hot_indices, is_gpu.tolist()).take)
        # # print('\n', i_layer, 'hot_indices:', hot_indices, 'is_gpu:', is_gpu)
        # # print(hot_experts, cold_experts)

        # for hot_expert, cold_expert in zip(hot_experts, cold_experts):
        #     if hot_indices[hot_expert] > swap_in_out * hot_indices[cold_expert] and is_cpu[hot_expert] and is_gpu[cold_expert]==1:
        #         assert not next(experts[hot_expert].parameters()).device == self.dev and next(experts[cold_expert].parameters()).device == self.dev
        #         # print(f'{i_layer}, swap {hot_expert, hot_indices[hot_expert]} to GPU; swap {cold_expert, hot_indices[cold_expert]} out')
        #         experts[cold_expert].to("cpu", non_blocking=True)
        #         experts[hot_expert].to(self.dev, non_blocking=True)

        routing_weights_dict_cpu, routing_weights_dict_gpu = {}, {}
        predict_experts_cpu, predict_experts_gpu = [], []
        # Determine which experts are on GPU and which are on CPU
        is_gpu = torch.tensor([next(predict_layer_experts[i_expert].parameters()).device == self.dev for i_expert in range(self.n_expert)])
        # Extract weights and indices for GPU and CPU experts
        if is_gpu.any() and (~is_gpu).any():
            top_weights_gpu, top_indices_gpu = torch.topk(predict_routing_weights[0][is_gpu], 2, dim=-1)
            top_weights_cpu, top_indices_cpu = torch.topk(predict_routing_weights[0][~is_gpu], 1, dim=-1)
        elif not is_gpu.any():
            top_weights_cpu, top_indices_cpu = torch.topk(predict_routing_weights[0][~is_gpu], 2, dim=-1)
            top_weights_gpu, top_indices_gpu = torch.tensor([]), torch.tensor([])
        elif not (~is_gpu).any():
            top_weights_gpu, top_indices_gpu = torch.topk(predict_routing_weights[0][is_gpu], 2, dim=-1)
            top_weights_cpu, top_indices_cpu = torch.tensor([]), torch.tensor([])

        # Convert indices to global indexing
        global_indices = torch.arange(self.n_expert).to(self.dev, non_blocking=True)
        global_indices_gpu = global_indices[is_gpu][top_indices_gpu]
        global_indices_cpu = global_indices[~is_gpu][top_indices_cpu]

        # print('is_gpu:', is_gpu)
        # print('predict_routing_weights:', predict_routing_weights)
        # print('top_experts_cpu:', global_indices_cpu, top_weights_cpu)
        # print('top_experts_gpu:', global_indices_gpu, top_weights_gpu)
        if not top_weights_gpu.any():
            top_weights_cpu /= top_weights_cpu.sum(dim=-1, keepdim=True)
            for index, expert_i in enumerate(global_indices_cpu):
                routing_weights_dict_cpu[int(expert_i)] = top_weights_cpu[index:index+1]

            # print('top_experts_cpu:', [], "top_weights_cpu:", {})
            # print('top_experts_gpu:', global_indices_gpu.tolist(), "top_weights_gpu:", routing_weights_dict_gpu)
            # input()
            return global_indices_cpu.tolist(), [], routing_weights_dict_cpu, {}
        # Comparison logic based on your requirements
        if (top_weights_gpu > top_weights_cpu).all():
            top_weights_gpu /= top_weights_gpu.sum(dim=-1, keepdim=True)
            for index, expert_i in enumerate(global_indices_gpu):
                routing_weights_dict_gpu[int(expert_i)] = top_weights_gpu[index:index+1]

            # print('top_experts_cpu:', [], "top_weights_cpu:", {})
            # print('top_experts_gpu:', global_indices_gpu.tolist(), "top_weights_gpu:", routing_weights_dict_gpu)
            # input()
            return [], global_indices_gpu.tolist(), {}, routing_weights_dict_gpu
        else:
            # Find the larger GPU expert if both are present
            if top_weights_gpu[0] > top_weights_gpu[1]:
                predict_experts_gpu.append(global_indices_gpu[0].item())
                top_weights_dict = torch.cat([top_weights_gpu[0:1], top_weights_cpu])
            else:
                predict_experts_gpu.append(global_indices_gpu[1].item())
                top_weights_dict = torch.cat([top_weights_gpu[1:2], top_weights_cpu])

            top_weights_dict /= top_weights_dict.sum(dim=-1, keepdim=True)
            routing_weights_dict_gpu[int(predict_experts_gpu[0])] = top_weights_dict[0:1]
            routing_weights_dict_cpu[int(global_indices_cpu[0])] = top_weights_dict[1:2]

            # print('top_experts_cpu:', global_indices_cpu.tolist(), "top_weights_cpu:", routing_weights_dict_cpu)
            # print('top_experts_gpu:', predict_experts_gpu, "top_weights_gpu:", routing_weights_dict_gpu)
            # input()
            return global_indices_cpu.tolist(), predict_experts_gpu, routing_weights_dict_cpu, routing_weights_dict_gpu
    

    def pre_run_expert_in_thread_cpu(self, hidden_states, index_predict_layer, pre_run_experts_cpu, pre_routing_weights_dict_cpu):
        predict_inps_after_experts_cpu = torch.zeros_like(hidden_states, device=self.dev)                      
        for expert_i in pre_run_experts_cpu:
            predict_inps_after_experts_cpu += self.run_expert_at_cpu(
                index_predict_layer,
                expert_i,
                hidden_states.to("cpu", non_blocking=True),
                pre_routing_weights_dict_cpu[expert_i].to("cpu", non_blocking=True),
            ).to(self.dev, non_blocking=True)
        return predict_inps_after_experts_cpu 



class Fast_PhiMoEModel(PhiMoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiMoEDecoderLayer`]

    Args:
        config: PhiMoEConfig
    """

    def __init__(self, model, config: PhiMoEConfig, _device):
        super().__init__(config)    
        self.padding_idx = model.padding_idx
        self.vocab_size = model.vocab_size
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self._attn_implementation = model._attn_implementation
        self.norm = model.norm
        self.gradient_checkpointing = model.gradient_checkpointing

        self.n_layer = config.num_hidden_layers
        self.n_expert = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # TODO: find this value based on device config
        self.executor = ThreadPoolExecutor(2)
        self.dev = _device

        self.expert_placeholder = copy.deepcopy(
            self.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        
        self.activate_matrix = np.zeros((self.n_layer, self.n_expert), dtype=int)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of PhiMoE. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        hidden_states, all_hidden_states, all_self_attns, all_router_logits, next_decoder_cache = self.PhiMoE_forward(hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            use_cache=use_cache,)
        
        # for decoder_layer in self.layers:
        #     if output_hidden_states:
        #         all_hidden_states += (hidden_states,)

        #     if self.gradient_checkpointing and self.training:
        #         layer_outputs = self._gradient_checkpointing_func(
        #             decoder_layer.__call__,
        #             hidden_states,
        #             attention_mask,
        #             position_ids,
        #             past_key_values,
        #             output_attentions,
        #             output_router_logits,
        #             use_cache,
        #         )
        #     else:
        #         layer_outputs = decoder_layer(
        #             hidden_states,
        #             attention_mask=attention_mask,
        #             position_ids=position_ids,
        #             past_key_value=past_key_values,
        #             output_attentions=output_attentions,
        #             output_router_logits=output_router_logits,
        #             use_cache=use_cache,
        #         )

        #     hidden_states = layer_outputs[0]

        #     if use_cache:
        #         next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        #     if output_attentions:
        #         all_self_attns += (layer_outputs[1],)

        #     if output_router_logits:
        #         all_router_logits += (layer_outputs[-1],)
                

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
    
    
    @torch.no_grad()
    def PhiMoE_forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        ########## added for expert pre-calculation ##########
        pre_calculated = {i:False for i in range(self.n_layer)}
        future_work_list = Queue()

        ### annotation
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # if sequence_length == 1:
        #     print('decode')
        # else:
        #     print('prefill')

        for i_layer, decoder_layer in enumerate(self.layers):

            ########## added for expert pre-calculation ##########
            i_predict_layer = min(i_layer+1, self.n_layer-1)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                )
            else:
                # layer_outputs = decoder_layer(
                #     hidden_states,
                #     attention_mask=attention_mask,
                #     position_ids=position_ids,
                #     past_key_value=past_key_value,
                #     output_attentions=output_attentions,
                #     output_router_logits=output_router_logits,
                #     use_cache=use_cache,
                # )
                
                residual = hidden_states

                hidden_states = decoder_layer.input_layernorm(hidden_states)

                # Self Attention
                hidden_states, self_attn_weights, present_key_value = decoder_layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states
                hidden_states = decoder_layer.post_attention_layernorm(hidden_states)

                # hidden_states, router_logits = decoder_layer.block_sparse_moe(hidden_states)

                batch_size, sequence_length, hidden_dim = hidden_states.shape

                is_decode = True if sequence_length == 1 else False
                experts = decoder_layer.block_sparse_moe.experts

                smoe = decoder_layer.block_sparse_moe
                if smoe.training and smoe.input_jitter_noise > 0:
                    hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - smoe.input_jitter_noise, 1.0 + smoe.input_jitter_noise)
                hidden_states = hidden_states.view(-1, hidden_dim)
                # # intermediate variable to store the output of experts
                final_hidden_states = torch.zeros(
                    (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=self.dev
                )
                final_hidden_states_cpu = torch.zeros(
                    (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=self.dev
                )

                if not is_decode:
                    router_logits = smoe.gate(hidden_states)
                    routing_weights, selected_experts = sparsemixer(
                        router_logits, 
                        top_k=smoe.top_k, 
                        jitter_eps=smoe.router_jitter_noise, 
                        training=smoe.training,
                    )
                else:
                    if i_layer < first_predict_layer: # 0, 1, 2, 3, 4
                        router_logits, experts_cpu, experts_gpu, routing_weights_cpu, routing_weights_gpu = self.assign_gate_on_inps(i_layer, hidden_states, False)
                    else: 
                        experts_gpu = predict_experts_gpu
                        experts_cpu, routing_weights_cpu = [], []
                        routing_weights_gpu = routing_weights_dict_gpu
                        router_logits = predict_router_logits

                    if i_layer >= first_predict_layer - 1 and i_layer < self.n_layer - 1: # 4-30
                        ####################### expert prediction using next layer's gate function #######################
                        predict_router_logits, predict_experts_cpu, predict_experts_gpu, routing_weights_dict_cpu, routing_weights_dict_gpu = self.assign_gate_on_inps(i_predict_layer, hidden_states, True)
                    ####### expert pre-calculation #######
                    if i_layer >= first_predict_layer - 1 and i_layer < self.n_layer - 1 and len(predict_experts_cpu) > 0:  # 4-30
                        pre_calculated[i_predict_layer] = True
                        future_work_list.put(self.executor.submit(self.pre_run_expert_in_thread_cpu, hidden_states, i_predict_layer, predict_experts_cpu, routing_weights_dict_cpu))      
                    else: 
                        pre_calculated[i_predict_layer] = False
        

                # dominated experts on GPU, expert pre-calculate on cpu
                if not is_decode:
                    expert_mask = torch.nn.functional.one_hot(
                        selected_experts, num_classes=self.n_expert
                    ).permute(2, 1, 0)              

                    # first, calculate the number of tokens for each expert
                    idxs, top_2s = [], []
                    cost_per_expert = np.zeros((self.n_expert, 2), dtype=float)  # 0: CPU, 1: GPU
                           
                    hot_indices = []
                    is_gpu = torch.tensor([1 if next(experts[i_expert].parameters()).device == self.dev else sequence_length for i_expert in range(self.n_expert)])
                    is_cpu = torch.tensor([0 if next(experts[i_expert].parameters()).device == self.dev else 1 for i_expert in range(self.n_expert)])
                    

                    for i_expert in range(self.n_expert):
                        idx, top_2 = torch.where(expert_mask[i_expert])
                        idxs.append(idx)
                        top_2s.append(top_2)  # the token indices to be processed on this expert
                        # expected latency at CPU: number of token * cost_at_cpu
                        # expected latency at GPU: cost_at_gpu (constant)
                        cost_per_expert[i_expert, 0] = top_2.shape[0] * latency_cpu
                        cost_per_expert[i_expert, 1] = latency_gpu
                        if next(experts[i_expert].parameters()).device == self.dev:
                            # if the expert is in GPU, the latency at GPU is approximately 0
                            cost_per_expert[i_expert, 1] = 0

                        hot_indices.append(top_2.shape[0])
                    

                    # hot_expert = int(np.argmax(np.multiply(hot_indices, is_cpu)))
                    # cold_expert = int(np.argmin(np.multiply(hot_indices, is_gpu)))
                    hot_experts = heapq.nlargest(largest_num, range(self.n_expert), np.multiply(hot_indices, is_cpu.tolist()).take)
                    cold_experts = heapq.nsmallest(largest_num, range(self.n_expert), np.multiply(hot_indices, is_gpu.tolist()).take)
                    # print('\n', i_layer, 'hot_indices:', hot_indices, 'is_gpu:', is_gpu)
                    # print(hot_experts, cold_experts)

                    for hot_expert, cold_expert in zip(hot_experts, cold_experts):
                        if hot_indices[hot_expert] > swap_in_out * hot_indices[cold_expert] and is_cpu[hot_expert] and is_gpu[cold_expert]==1:
                            assert not next(experts[hot_expert].parameters()).device == self.dev and next(experts[cold_expert].parameters()).device == self.dev

                            # print(f'{i_layer}, swap {hot_expert, hot_indices[hot_expert]} to GPU; swap {cold_expert, hot_indices[cold_expert]} out')

                            experts[cold_expert].to("cpu", non_blocking=True)
                            experts[hot_expert].to(self.dev, non_blocking=True)
                            cost_per_expert[cold_expert, 1] = latency_gpu
                            cost_per_expert[hot_expert, 1] = 0

                            assert next(experts[hot_expert].parameters()).device == self.dev and not next(experts[cold_expert].parameters()).device == self.dev
                    
                    # second, partition experts processing between CPU and GPU so that we can minimize:
                    # max(sum of cost at CPU, sum of cost at GPU)
                    # greedy algorithm is just as there are only self.n_expert experts for PhiMoE
                    best_config = -1
                    best_cost = float("inf")
                    for config in range(1 << self.n_expert):
                        sum_cost = 0
                        for i_expert in range(self.n_expert):
                            if (config >> i_expert) & 1:
                                sum_cost += cost_per_expert[i_expert, 0]
                            else:
                                sum_cost += cost_per_expert[i_expert, 1]
                        
                        if sum_cost < best_cost:
                            best_cost = sum_cost
                            best_config = config

                    # then, we can offload the experts according to the best configuration
                    cpu_experts = []
                    gpu_experts = []
                    for i_expert in range(self.n_expert):
                        if (best_config >> i_expert) & 1:
                            cpu_experts.append(i_expert)
                        else:
                            gpu_experts.append(i_expert)

                    def run_expert_in_thread():
                        for i_expert in cpu_experts:
                            top_2_list = top_2s[i_expert].tolist()
                            idx_list = idxs[i_expert].tolist()
                            current_state = hidden_states[None, top_2_list].reshape(-1, hidden_dim)
                            current_state = self.run_expert_at_cpu(
                                i_layer,
                                i_expert,
                                current_state.to("cpu", non_blocking=True),
                                routing_weights[top_2_list, idx_list, None].to("cpu", non_blocking=True),
                            )
                            final_hidden_states_cpu.index_add_(
                                0,
                                top_2s[i_expert].to(self.dev, non_blocking=True),
                                current_state.to(self.dev, non_blocking=True),
                            )

                    if len(cpu_experts) > 0:
                        work_obj = [self.executor.submit(run_expert_in_thread)]

                    for i_expert in gpu_experts:
                        top_2_list = top_2s[i_expert].tolist()
                        idx_list = idxs[i_expert].tolist()
                        current_state = hidden_states[None, top_2_list].reshape(-1, hidden_dim)
                        if next(experts[i_expert].parameters()).device == self.dev:
                            # current_state = experts[i_expert](
                            #     current_state, routing_weights[top_2_list, idx_list, None]
                            # )
                            current_state = experts[i_expert](current_state) * routing_weights[top_2_list, idx_list, None]
                            
                        else:
                            self.expert_placeholder.load_state_dict(
                                experts[i_expert].state_dict()
                            )
                            # current_state = self.expert_placeholder(
                            #     current_state, routing_weights[top_2_list, idx_list, None]
                            # )
                            current_state = self.expert_placeholder(current_state) * routing_weights[top_2_list, idx_list, None]
                            
                            
                            # cold_expert = int(np.argmin(np.multiply(hot_indices, is_gpu)))
                            # is_gpu[cold_expert] = 1000
                            # is_gpu[i_expert] = 1
                            # experts[cold_expert].to("cpu", non_blocking=True)
                            # experts[i_expert].to(self.dev, non_blocking=True)
                            # current_state = experts[i_expert](
                            #     current_state, routing_weights[top_2_list, idx_list, None]
                            # )

                            # print('is_gpu:', is_gpu)
                            # print('swap', i_expert, 'to GPU; swap', cold_expert, 'out')
                            # print('i_expert:', i_expert, 'tokens:', hot_indices[i_expert], 'cold_expert:', cold_expert, 'tokens:', hot_indices[cold_expert])
                            # torch.cuda.synchronize()
                            # print('verification:', next(experts[i_expert].parameters()).device == self.dev, not next(experts[cold_expert].parameters()).device == self.dev)


                        final_hidden_states.index_add_(
                            0,
                            top_2s[i_expert].to(self.dev, non_blocking=True),
                            current_state.to(self.dev, non_blocking=True),
                        )
                        
                    if len(cpu_experts) > 0:  # gpu+cpu, cpu
                        wait(fs=work_obj, timeout=None, return_when=ALL_COMPLETED)
                        final_hidden_states += final_hidden_states_cpu
                        
                else:
                    # ####### expert pre-calculation #######
                    # if i_layer > 0 and i_layer < self.n_layer - 1 and len(predict_experts_cpu) > 0:  # 1-30
                    #     future_work_list.put(self.executor.submit(self.pre_run_expert_in_thread_cpu, inps, i_predict_layer, predict_experts_cpu, routing_weights_dict_cpu))  
                    #     pre_calculated[i_predict_layer] = True
                    # else: 
                    #     pre_calculated[i_predict_layer] = False
                    ####### decode stage with offloading -- current layer expert execution #######
                    for expert_i in experts_cpu:                    
                        final_hidden_states += self.run_expert_at_cpu(
                            i_layer,
                            expert_i,
                            hidden_states.to("cpu", non_blocking=True),
                            routing_weights_cpu[expert_i].to("cpu", non_blocking=True),
                        ).to(self.dev, non_blocking=True)
                    
                    for expert_i in experts_gpu:
                        # final_hidden_states += experts[expert_i](hidden_states, routing_weights_gpu[expert_i])
                        final_hidden_states += experts[expert_i](hidden_states) * routing_weights_gpu[expert_i]

                    ####### Receive expert pre-calculation results from CPU #######
                    if i_layer >= first_predict_layer and pre_calculated[i_layer]: # 4-30
                        done, not_done = wait(fs=[future_work_list.get()], timeout=None, return_when=FIRST_COMPLETED)
                        final_hidden_states_cpu = done.pop().result()
                        final_hidden_states += final_hidden_states_cpu.to(self.dev, non_blocking=False)
                        pre_calculated[i_layer] = False

                hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

                hidden_states = residual + hidden_states
                layer_outputs = (hidden_states,)

                if output_attentions:
                    layer_outputs += (self_attn_weights,)

                if use_cache:
                    layer_outputs += (present_key_value,)

                if output_router_logits:
                    layer_outputs += (router_logits,)

                # addition because there's residual connection over moe layer
                # end of one layer


            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

    
        return hidden_states, all_hidden_states, all_self_attns, all_router_logits, next_decoder_cache


    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        # return self.layers[i_layer].block_sparse_moe.experts[i_expert](
        #     inps, routing_weights
        # )
        return self.layers[i_layer].block_sparse_moe.experts[i_expert](inps) * routing_weights
        
    

    def assign_gate_on_inps(self, i_predict_layer, hidden_states, pre_calulation):
        smoe = self.layers[i_predict_layer].block_sparse_moe
        predict_layer_experts = smoe.experts # 1 - 31
        # batch_size, sequence_length, hidden_dim = hidden_states.shape

        # predict_router_logits = smoe.gate(hidden_states)
        # predict_routing_weights = F.softmax(predict_router_logits, dim=1)
        # top_weights, top_experts = torch.topk(predict_routing_weights, 2, dim=-1, sorted=True)
        # top_weights /= top_weights.sum(dim=-1, keepdim=True)

        # if smoe.training and smoe.input_jitter_noise > 0:
        #     hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - smoe.input_jitter_noise, 1.0 + smoe.input_jitter_noise)
        # hidden_states = hidden_states.view(-1, hidden_dim)
        predict_router_logits = smoe.gate(hidden_states)

        predict_routing_weights = F.softmax(predict_router_logits, dim=1)

        top_weights, top_experts = sparsemixer(
            predict_router_logits, 
            top_k=smoe.top_k, 
            jitter_eps=smoe.router_jitter_noise, 
            training=smoe.training,
        )
        # print('predict_routing_weights:', predict_routing_weights, 'top_weights:', top_weights, 'top_experts:', top_experts)
        # # print('before:', top_experts, top_weights) #tensor([[2, 4]], device='cuda') tensor([[0.6992, 0.2988]], device='cuda', dtype=torch.bfloat16)
        # input()

        routing_weights_dict_cpu, routing_weights_dict_gpu = {}, {}
        predict_experts_cpu, predict_experts_gpu = [], []
        for index in range(len(top_experts[0])):
            i_expert = int(top_experts[0][index])
            if next(predict_layer_experts[i_expert].parameters()).device == self.dev:
                predict_experts_gpu.append(i_expert)
                routing_weights_dict_gpu[i_expert] = top_weights[:, index, None]
                ### added by yujie
                self.activate_matrix[i_predict_layer][i_expert] += 1
            else:
                predict_experts_cpu.append(i_expert)
                routing_weights_dict_cpu[i_expert] = top_weights[:, index, None]
                ### added by yujie
                self.activate_matrix[i_predict_layer][i_expert] += 1

        # if two experts are all on cpu, then select one most similar expert on gpu as an alternative option
        if len(predict_experts_cpu) == 2 and pre_calulation:
            is_gpu = torch.tensor([next(predict_layer_experts[i_expert].parameters()).device == self.dev for i_expert in range(self.n_expert)])
            # print(i_predict_layer, 'is_gpu:', is_gpu, predict_routing_weights, predict_experts_cpu)
            if is_gpu.any():
                
                # hot_expert = predict_experts_cpu[0]

                # # last_weights_cpu, last_indices_gpu = torch.topk(predict_routing_weights[0][is_gpu].flip(dims=[0]), 1, dim=-1)
                # # global_indices = torch.arange(self.n_expert).to(self.dev, non_blocking=True)
                # # global_indices_gpu = global_indices[is_gpu][last_indices_gpu]
                # # cold_expert = global_indices_gpu[0].item()

                # last_indices_gpu = np.argmin(self.activate_matrix[i_predict_layer][is_gpu])
                # global_indices = torch.arange(self.n_expert).to(self.dev, non_blocking=True)
                # global_indices_gpu = global_indices[is_gpu][torch.tensor((last_indices_gpu))]
                # cold_expert = global_indices_gpu.item()
                
                # # print(i_predict_layer, 'is_gpu:', is_gpu, predict_routing_weights, predict_experts_cpu, self.activate_matrix[i_predict_layer][is_gpu], last_indices_gpu, global_indices_gpu)

                popped_expert = predict_experts_cpu.pop()

                # if self.activate_matrix[i_predict_layer][hot_expert] >  self.activate_matrix[i_predict_layer][cold_expert] and self.activate_matrix[i_predict_layer][cold_expert] / sum(self.activate_matrix[i_predict_layer]) < swap_in_out - 1:
                #     assert not next(predict_layer_experts[hot_expert].parameters()).device == self.dev and next(predict_layer_experts[cold_expert].parameters()).device == self.dev     
                    
                #     predict_layer_experts[cold_expert].to("cpu", non_blocking=True)
                #     predict_layer_experts[hot_expert].to(self.dev, non_blocking=True)
                #     predict_experts_cpu[0] = popped_expert
                #     predict_experts_gpu.append(hot_expert)
                #     routing_weights_dict_gpu[hot_expert] = routing_weights_dict_cpu[hot_expert]
                #     routing_weights_dict_cpu.pop(hot_expert, None)

                #     assert next(predict_layer_experts[hot_expert].parameters()).device == self.dev and not next(predict_layer_experts[cold_expert].parameters()).device == self.dev
                #     # print(f'{i_predict_layer}, swap {hot_expert, predict_routing_weights[0][hot_expert]} to GPU; swap {cold_expert, predict_routing_weights[0][cold_expert]} out')
                # else:
                top_weights_gpu, top_indices_gpu = torch.topk(predict_routing_weights[0][is_gpu], 1, dim=-1)
                global_indices = torch.arange(self.n_expert).to(self.dev, non_blocking=True)
                global_indices_gpu = global_indices[is_gpu][top_indices_gpu]
                predict_expert_index_gpu = global_indices_gpu[0].item()
                # print(predict_routing_weights, predict_experts_cpu, is_gpu, top_weights_gpu)
                
                # if predict_routing_weights[0][predict_experts_cpu[1]] - min(top_weights_gpu) < 0.05:
                #     # Find the most similar GPU expert if both are present
                #     routing_weights_dict_cpu.pop(predict_experts_cpu.pop(), None)
                #     predict_expert_index_cpu = predict_experts_cpu[0]
                #     predict_experts_gpu.append(predict_expert_index_gpu)

                #     top_weights_dict = torch.tensor([[predict_routing_weights[0][predict_expert_index_gpu], predict_routing_weights[0][predict_expert_index_cpu]]]).to(self.dev, non_blocking=True)
                #     top_weights_dict /= top_weights_dict.sum(dim=-1, keepdim=True)
                #     routing_weights_dict_gpu[predict_expert_index_gpu] = top_weights_dict[:, 0, None]
                #     routing_weights_dict_cpu[predict_expert_index_cpu] = top_weights_dict[:, 1, None]

                # Find the larger CPU expert if both are present
                # popped_expert = predict_experts_cpu.pop()
                routing_weights_dict_cpu.pop(popped_expert, None)
                predict_expert_index_cpu = predict_experts_cpu[0]
                predict_experts_gpu.append(predict_expert_index_gpu)
                ### added by yujie
                self.activate_matrix[i_predict_layer][predict_expert_index_gpu] += 1
                self.activate_matrix[i_predict_layer][popped_expert] -= 1

                top_weights_dict = torch.tensor([[predict_routing_weights[0][predict_expert_index_gpu], predict_routing_weights[0][predict_expert_index_cpu]]]).to(self.dev, non_blocking=True)
                top_weights_dict /= top_weights_dict.sum(dim=-1, keepdim=True)
                routing_weights_dict_gpu[predict_expert_index_gpu] = top_weights_dict[:, 0, None]
                routing_weights_dict_cpu[predict_expert_index_cpu] = top_weights_dict[:, 1, None]


        return predict_router_logits, predict_experts_cpu, predict_experts_gpu, routing_weights_dict_cpu, routing_weights_dict_gpu
    
        # hot_indices = []
        # is_gpu = torch.tensor([1 if next(experts[i_expert].parameters()).device == self.dev else sequence_length for i_expert in range(self.n_expert)])
        # is_cpu = torch.tensor([0 if next(experts[i_expert].parameters()).device == self.dev else 1 for i_expert in range(self.n_expert)])
        
        # # hot_expert = int(np.argmax(np.multiply(hot_indices, is_cpu)))
        # # cold_expert = int(np.argmin(np.multiply(hot_indices, is_gpu)))
        # hot_experts = heapq.nlargest(largest_num, range(self.n_expert), np.multiply(hot_indices, is_cpu.tolist()).take)
        # cold_experts = heapq.nsmallest(largest_num, range(self.n_expert), np.multiply(hot_indices, is_gpu.tolist()).take)
        # # print('\n', i_layer, 'hot_indices:', hot_indices, 'is_gpu:', is_gpu)
        # # print(hot_experts, cold_experts)

        # for hot_expert, cold_expert in zip(hot_experts, cold_experts):
        #     if hot_indices[hot_expert] > swap_in_out * hot_indices[cold_expert] and is_cpu[hot_expert] and is_gpu[cold_expert]==1:
        #         assert not next(experts[hot_expert].parameters()).device == self.dev and next(experts[cold_expert].parameters()).device == self.dev
        #         # print(f'{i_layer}, swap {hot_expert, hot_indices[hot_expert]} to GPU; swap {cold_expert, hot_indices[cold_expert]} out')
        #         experts[cold_expert].to("cpu", non_blocking=True)
        #         experts[hot_expert].to(self.dev, non_blocking=True)

        routing_weights_dict_cpu, routing_weights_dict_gpu = {}, {}
        predict_experts_cpu, predict_experts_gpu = [], []
        # Determine which experts are on GPU and which are on CPU
        is_gpu = torch.tensor([next(predict_layer_experts[i_expert].parameters()).device == self.dev for i_expert in range(self.n_expert)])
        # Extract weights and indices for GPU and CPU experts
        if is_gpu.any() and (~is_gpu).any():
            top_weights_gpu, top_indices_gpu = torch.topk(predict_routing_weights[0][is_gpu], 2, dim=-1)
            top_weights_cpu, top_indices_cpu = torch.topk(predict_routing_weights[0][~is_gpu], 1, dim=-1)
        elif not is_gpu.any():
            top_weights_cpu, top_indices_cpu = torch.topk(predict_routing_weights[0][~is_gpu], 2, dim=-1)
            top_weights_gpu, top_indices_gpu = torch.tensor([]), torch.tensor([])
        elif not (~is_gpu).any():
            top_weights_gpu, top_indices_gpu = torch.topk(predict_routing_weights[0][is_gpu], 2, dim=-1)
            top_weights_cpu, top_indices_cpu = torch.tensor([]), torch.tensor([])

        # Convert indices to global indexing
        global_indices = torch.arange(self.n_expert).to(self.dev, non_blocking=True)
        global_indices_gpu = global_indices[is_gpu][top_indices_gpu]
        global_indices_cpu = global_indices[~is_gpu][top_indices_cpu]

        # print('is_gpu:', is_gpu)
        # print('predict_routing_weights:', predict_routing_weights)
        # print('top_experts_cpu:', global_indices_cpu, top_weights_cpu)
        # print('top_experts_gpu:', global_indices_gpu, top_weights_gpu)
        if not top_weights_gpu.any():
            top_weights_cpu /= top_weights_cpu.sum(dim=-1, keepdim=True)
            for index, expert_i in enumerate(global_indices_cpu):
                routing_weights_dict_cpu[int(expert_i)] = top_weights_cpu[index:index+1]

            # print('top_experts_cpu:', [], "top_weights_cpu:", {})
            # print('top_experts_gpu:', global_indices_gpu.tolist(), "top_weights_gpu:", routing_weights_dict_gpu)
            # input()
            return global_indices_cpu.tolist(), [], routing_weights_dict_cpu, {}
        # Comparison logic based on your requirements
        if (top_weights_gpu > top_weights_cpu).all():
            top_weights_gpu /= top_weights_gpu.sum(dim=-1, keepdim=True)
            for index, expert_i in enumerate(global_indices_gpu):
                routing_weights_dict_gpu[int(expert_i)] = top_weights_gpu[index:index+1]

            # print('top_experts_cpu:', [], "top_weights_cpu:", {})
            # print('top_experts_gpu:', global_indices_gpu.tolist(), "top_weights_gpu:", routing_weights_dict_gpu)
            # input()
            return [], global_indices_gpu.tolist(), {}, routing_weights_dict_gpu
        else:
            # Find the larger GPU expert if both are present
            if top_weights_gpu[0] > top_weights_gpu[1]:
                predict_experts_gpu.append(global_indices_gpu[0].item())
                top_weights_dict = torch.cat([top_weights_gpu[0:1], top_weights_cpu])
            else:
                predict_experts_gpu.append(global_indices_gpu[1].item())
                top_weights_dict = torch.cat([top_weights_gpu[1:2], top_weights_cpu])

            top_weights_dict /= top_weights_dict.sum(dim=-1, keepdim=True)
            routing_weights_dict_gpu[int(predict_experts_gpu[0])] = top_weights_dict[0:1]
            routing_weights_dict_cpu[int(global_indices_cpu[0])] = top_weights_dict[1:2]

            # print('top_experts_cpu:', global_indices_cpu.tolist(), "top_weights_cpu:", routing_weights_dict_cpu)
            # print('top_experts_gpu:', predict_experts_gpu, "top_weights_gpu:", routing_weights_dict_gpu)
            # input()
            return global_indices_cpu.tolist(), predict_experts_gpu, routing_weights_dict_cpu, routing_weights_dict_gpu
    

    def pre_run_expert_in_thread_cpu(self, hidden_states, index_predict_layer, pre_run_experts_cpu, pre_routing_weights_dict_cpu):
        predict_inps_after_experts_cpu = torch.zeros_like(hidden_states, device=self.dev)                      
        for expert_i in pre_run_experts_cpu:
            predict_inps_after_experts_cpu += self.run_expert_at_cpu(
                index_predict_layer,
                expert_i,
                hidden_states.to("cpu", non_blocking=True),
                pre_routing_weights_dict_cpu[expert_i].to("cpu", non_blocking=True),
            ).to(self.dev, non_blocking=True)
        return predict_inps_after_experts_cpu 


