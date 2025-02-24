import warnings
from typing import Optional, Tuple, Union, List
import time
import numpy as np
import torch
import torch.nn as nn
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
from data import latency_cpu, latency_gpu

def get_kv_cache_size(past_key_value):
    total_size = 0
    for layer_past in past_key_value:# Each layer_past is a tuple of (key, value)
        for tensor in layer_past:
            total_size += tensor.element_size() * tensor.nelement()
    return total_size

def format_size(size_in_bytes):# Format the size to display in bytes, KB, and MB
    kb = size_in_bytes / 1024
    mb = kb / 1024
    return f"{mb:.2f} MB"


class Fiddler_MixtralModel(MixtralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MixtralDecoderLayer`]

    Args:
        config: MixtralConfig
    """

    def __init__(self, model, config: MixtralConfig, cpu_offload):
        super().__init__(config)
    
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
        self.dev = torch.device("cuda:0")
        self.cpu_offload = cpu_offload

        self.expert_placeholder = copy.deepcopy(
            self.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)

        self.num_generated_tokens = 0
        self.term_num_list = [64, 256, 1024, 4096, 16384]
        self.kv_cache = {'prefill': '', **{i: '' for i in self.term_num_list}}

    def initialize_info(self):
        self.num_generated_tokens = 0
        self.kv_cache = {'prefill': '', **{i: '' for i in self.term_num_list}}

    def debug_printout(self):
        print(f"""kv_cache: {self.num_generated_tokens}, {self.term_num_list}, {self.kv_cache}""")

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

        _, seq_len, _ = hidden_states.shape
        is_decode = True if seq_len == 1 else False
        if is_decode == False:
            self.num_generated_tokens = 0 

        for i_layer, decoder_layer in enumerate(self.layers):

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


                # is_decode = True if sequence_length == 1 else False
                experts = decoder_layer.block_sparse_moe.experts

                # # intermediate variable to store the output of experts
                final_hidden_states = torch.zeros(
                    (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=self.dev
                )
                final_hidden_states_cpu = torch.zeros(
                    (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=self.dev
                )

                router_logits = decoder_layer.block_sparse_moe.gate(hidden_states)
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                # we cast back to the input dtype
                routing_weights = routing_weights.to(hidden_states.dtype)
                

                if self.cpu_offload == 0:
                    # baseline: do everything at GPU
                    
                    expert_mask = torch.nn.functional.one_hot(
                        selected_experts, num_classes=self.n_expert
                    ).permute(2, 1, 0)

                    for i_expert in range(self.n_expert):
                        is_cuda = (next(experts[i_expert].parameters()).device == self.dev)
                        idx, top_2 = torch.where(expert_mask[i_expert])

                        if top_2.shape[0] == 0:
                            # print(f"Expert {i_expert}: has no tokens")
                            continue

                        top_2_list = top_2.tolist()
                        idx_list = idx.tolist()

                        current_state = hidden_states[None, top_2_list].reshape(-1, hidden_dim)
                        if not is_cuda:
                            self.expert_placeholder.load_state_dict(
                                experts[i_expert].state_dict()
                            )
                            current_state = self.expert_placeholder(current_state) * routing_weights[top_2_list, idx_list, None]
                        else:
                            current_state = experts[i_expert](current_state) * routing_weights[top_2_list, idx_list, None]
                        final_hidden_states.index_add_(
                            0, top_2, current_state.to(hidden_states.dtype)
                        )

                        if not is_cuda:
                            experts[i_expert] = experts[i_expert].to("cpu", non_blocking=True)
                        # end of one expert

                elif not is_decode:

                    expert_mask = torch.nn.functional.one_hot(
                        selected_experts, num_classes=self.n_expert
                    ).permute(2, 1, 0)              

                    # first, calculate the number of tokens for each expert
                    idxs, top_2s = [], []
                    cost_per_expert = np.zeros((self.n_expert, 2), dtype=float)  # 0: CPU, 1: GPU

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
                    for i_expert in range(8):
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
                        final_hidden_states.index_add_(
                            0,
                            top_2s[i_expert].to(self.dev, non_blocking=True),
                            current_state.to(self.dev, non_blocking=True),
                        )
                        
                    if len(cpu_experts) > 0:  # gpu+cpu, cpu
                        wait(fs=work_obj, timeout=None, return_when=ALL_COMPLETED)
                        final_hidden_states += final_hidden_states_cpu
                        
                else:
                    # decode stage with offloading
                    expert_0, expert_1 = int(selected_experts[0][0]), int(
                        selected_experts[0][1]
                    )

                    routing_weights_0, routing_weights_1 = (
                        routing_weights[:, 0, None],
                        routing_weights[:, 1, None],
                    )
                    assert expert_0 != expert_1

                    if next(experts[expert_0].parameters()).device == self.dev:
                        # final_hidden_states += experts[expert_0](hidden_states, routing_weights_0)
                        final_hidden_states += experts[expert_0](hidden_states) * routing_weights_0
                    else:
                        final_hidden_states += self.run_expert_at_cpu(
                            i_layer,
                            expert_0,
                            hidden_states.to("cpu", non_blocking=True),
                            routing_weights_0.to("cpu", non_blocking=True),
                        ).to(self.dev, non_blocking=True)

                    if next(experts[expert_1].parameters()).device == self.dev:
                        # final_hidden_states += experts[expert_1](hidden_states, routing_weights_1)
                        final_hidden_states += experts[expert_1](hidden_states) * routing_weights_1
                    else:
                        final_hidden_states += self.run_expert_at_cpu(
                            i_layer,
                            expert_1,
                            hidden_states.to("cpu", non_blocking=True),
                            routing_weights_1.to("cpu", non_blocking=True),
                        ).to(self.dev, non_blocking=True)

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

        self.num_generated_tokens = self.num_generated_tokens + 1
        # self.record_kv_cache_size(is_decode, past_key_value)                
        return hidden_states, all_hidden_states, all_self_attns, all_router_logits, next_decoder_cache


    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        return self.layers[i_layer].block_sparse_moe.experts[i_expert](inps) * routing_weights
    

    def record_kv_cache_size(self, is_decode, past_key_value):
        if is_decode == False:            
            kv_cache_size = get_kv_cache_size(past_key_value)
            formatted_size = format_size(kv_cache_size)
            self.kv_cache['prefill'] = formatted_size
        elif self.num_generated_tokens in self.term_num_list:
            kv_cache_size = get_kv_cache_size(past_key_value)
            formatted_size = format_size(kv_cache_size)
            self.kv_cache[self.num_generated_tokens] = formatted_size


class Fiddler_PhiMoEModel(PhiMoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiMoEDecoderLayer`]

    Args:
        config: PhiMoEConfig
    """

    def __init__(self, model, config: PhiMoEConfig, cpu_offload):
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
        self.dev = torch.device("cuda:0")
        self.cpu_offload = cpu_offload

        self.expert_placeholder = copy.deepcopy(
            self.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)

        self.num_generated_tokens = 0
        self.term_num_list = [64, 256, 1024, 4096, 16384]
        self.kv_cache = {'prefill': '', **{i: '' for i in self.term_num_list}}

    def initialize_info(self):
        self.num_generated_tokens = 0
        self.kv_cache = {'prefill': '', **{i: '' for i in self.term_num_list}}

    def debug_printout(self):
        print(f"""kv_cache: {self.num_generated_tokens}, {self.term_num_list}, {self.kv_cache}""")
        
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

        _, seq_len, _ = hidden_states.shape
        is_decode = True if seq_len == 1 else False
        if is_decode == False:
            self.num_generated_tokens = 0 
            
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
                # is_decode = True if sequence_length == 1 else False
                experts = decoder_layer.block_sparse_moe.experts

                smoe = decoder_layer.block_sparse_moe
                if smoe.training and smoe.input_jitter_noise > 0:
                    hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - smoe.input_jitter_noise, 1.0 + smoe.input_jitter_noise)
                hidden_states = hidden_states.view(-1, hidden_dim)
                router_logits = smoe.gate(hidden_states)
                routing_weights, selected_experts = sparsemixer(
                    router_logits, 
                    top_k=smoe.top_k, 
                    jitter_eps=smoe.router_jitter_noise, 
                    training=smoe.training,
                )

                # print('router_logits:', router_logits, 'routing_weights:', routing_weights, 'selected_experts:', selected_experts)
                # input()

                # # intermediate variable to store the output of experts
                final_hidden_states = torch.zeros(
                    (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=self.dev
                )
                final_hidden_states_cpu = torch.zeros(
                    (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=self.dev
                )
                

                if self.cpu_offload == 0:
                    # baseline: do everything at GPU
                    
                    expert_mask = torch.nn.functional.one_hot(
                        selected_experts, num_classes=self.n_expert
                    ).permute(2, 1, 0)

                    for i_expert in range(self.n_expert):
                        is_cuda = (next(experts[i_expert].parameters()).device == self.dev)
                        idx, top_2 = torch.where(expert_mask[i_expert])

                        if top_2.shape[0] == 0:
                            # print(f"Expert {i_expert}: has no tokens")
                            continue

                        top_2_list = top_2.tolist()
                        idx_list = idx.tolist()

                        current_state = hidden_states[None, top_2_list].reshape(-1, hidden_dim)
                        if not is_cuda:
                            self.expert_placeholder.load_state_dict(
                                experts[i_expert].state_dict()
                            )
                            current_state = self.expert_placeholder(current_state) * routing_weights[top_2_list, idx_list, None]
                        else:
                            current_state = experts[i_expert](current_state) * routing_weights[top_2_list, idx_list, None]
                        final_hidden_states.index_add_(
                            0, top_2, current_state.to(hidden_states.dtype)
                        )

                        if not is_cuda:
                            experts[i_expert] = experts[i_expert].to("cpu", non_blocking=True)
                        # end of one expert

                elif not is_decode:

                    expert_mask = torch.nn.functional.one_hot(
                        selected_experts, num_classes=self.n_expert
                    ).permute(2, 1, 0)              

                    # first, calculate the number of tokens for each expert
                    idxs, top_2s = [], []
                    cost_per_expert = np.zeros((self.n_expert, 2), dtype=float)  # 0: CPU, 1: GPU

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
                        final_hidden_states.index_add_(
                            0,
                            top_2s[i_expert].to(self.dev, non_blocking=True),
                            current_state.to(self.dev, non_blocking=True),
                        )
                        
                    if len(cpu_experts) > 0:  # gpu+cpu, cpu
                        wait(fs=work_obj, timeout=None, return_when=ALL_COMPLETED)
                        final_hidden_states += final_hidden_states_cpu
                        
                else:
                    # decode stage with offloading
                    expert_0, expert_1 = int(selected_experts[0][0]), int(
                        selected_experts[0][1]
                    )

                    routing_weights_0, routing_weights_1 = (
                        routing_weights[:, 0, None],
                        routing_weights[:, 1, None],
                    )
                    assert expert_0 != expert_1
                    
                    if next(experts[expert_0].parameters()).device == self.dev:
                        # final_hidden_states += experts[expert_0](hidden_states, routing_weights_0)
                        final_hidden_states += experts[expert_0](hidden_states) * routing_weights_0
                    else:
                        final_hidden_states += self.run_expert_at_cpu(
                            i_layer,
                            expert_0,
                            hidden_states.to("cpu", non_blocking=True),
                            routing_weights_0.to("cpu", non_blocking=True),
                        ).to(self.dev, non_blocking=True)

                    if next(experts[expert_1].parameters()).device == self.dev:
                        # final_hidden_states += experts[expert_1](hidden_states, routing_weights_1)
                        final_hidden_states += experts[expert_1](hidden_states) * routing_weights_1
                    else:
                        final_hidden_states += self.run_expert_at_cpu(
                            i_layer,
                            expert_1,
                            hidden_states.to("cpu", non_blocking=True),
                            routing_weights_1.to("cpu", non_blocking=True),
                        ).to(self.dev, non_blocking=True)

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

        self.num_generated_tokens = self.num_generated_tokens + 1
        # self.record_kv_cache_size(is_decode, past_key_value)                
        return hidden_states, all_hidden_states, all_self_attns, all_router_logits, next_decoder_cache


    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        return self.layers[i_layer].block_sparse_moe.experts[i_expert](inps) * routing_weights
    

    def record_kv_cache_size(self, is_decode, past_key_value):
        if is_decode == False:            
            kv_cache_size = get_kv_cache_size(past_key_value)
            formatted_size = format_size(kv_cache_size)
            self.kv_cache['prefill'] = formatted_size
        elif self.num_generated_tokens in self.term_num_list:
            kv_cache_size = get_kv_cache_size(past_key_value)
            formatted_size = format_size(kv_cache_size)
            self.kv_cache[self.num_generated_tokens] = formatted_size
        


