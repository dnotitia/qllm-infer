from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing import Callable
from lib.quantization.quantizer import *
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 이후 logger 사용
logger.info("Some message")

def opt_attn_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # isn't needed in normal attention, but needed in flash attention so to keep the signature same
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        attn_weights = self.Query_Key_matmul(query_states, key_states.transpose(3, 2))
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = self.Softmax_Value_matmul(attn_probs, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs, past_key_value


def llama_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    #print("new eager attention")
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    #attn_weights = module.Query_Key_matmul(query, key_states.transpose(2, 3)) * scaling
    if hasattr(module, "Query_Key_matmul"):
        attn_weights = module.Query_Key_matmul(query, key_states.transpose(-2, -1)) * scaling
    else:
        attn_weights = torch.matmul(query, key_states.transpose(-2, -1)) * scaling
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    #attn_output = module.Softmax_Value_matmul(attn_weights, value_states)
    if hasattr(module, "Softmax_Value_matmul"):
        attn_output = module.Softmax_Value_matmul(attn_weights, value_states)
    else:
        attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



def gemma3_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if hasattr(module, "Query_Key_matmul"):
        attn_weights = module.Query_Key_matmul(query, key_states.transpose(2, 3)) * scaling
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if hasattr(module, "Softmax_Value_matmul"):
        attn_output = module.Softmax_Value_matmul(attn_weights, value_states)
    else:
        attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def qwen3_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if hasattr(module, "Query_Key_matmul"):
        attn_weights = module.Query_Key_matmul(query, key_states.transpose(2, 3)) * scaling
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if hasattr(module, "Softmax_Value_matmul"):
        attn_output = module.Softmax_Value_matmul(attn_weights, value_states)
    else:
        attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def llama_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        #qt=Quantizer()
        #qt.configure(
        #    bits=8, r_bit=4, r_group=32, qrazor=True, perchannel=True, sym=False, mse=False
        #)
        #key_states_shape_ = key_states.shape
        #key_states = key_states.reshape(-1, key_states_shape_[-1])
        #qt.find_params(key_states, weight=True)
        #key_states = quantize(
        #    key_states,
        #    qt.scale, qt.zero, qt.maxq,
        #    qt.sym, qt.r_bit, qt.r_group,
        #    qt.bits, qt.qrazor
        #).to(key_states.dtype).reshape(key_states_shape_)

        if self.config.k_pre_RoPE_quant:
            #print("in pre RoPE quant")
            qt=Quantizer()
            qt.configure(
                self.config.bits_k, 
                self.config.k_qrazor_bits, 
                self.config.k_qrazor_group, 
                self.config.k_qrazor, 
                not self.config.k_perchannel, 
                self.config.sym_k, 
                mse=False
            )
            key_states_shape_ = key_states.shape
            key_states = key_states.reshape(-1, key_states_shape_[-1])
            qt.find_params(key_states, weight=True)
            key_states = quantize(
                key_states,
                qt.scale, qt.zero, qt.maxq,
                qt.sym, qt.r_bit, qt.r_group,
                qt.bits, qt.qrazor
            ).to(key_states.dtype).reshape(key_states_shape_)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = llama_eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def gemma3_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        #qt=Quantizer()
        #qt.configure(
        #    bits=8, r_bit=4, r_group=32, qrazor=True, perchannel=True, sym=False, mse=False
        #)
        #key_states_shape_ = key_states.shape
        #key_states = key_states.reshape(-1, key_states_shape_[-1])
        #qt.find_params(key_states, weight=True)
        #key_states = quantize(
        #    key_states,
        #    qt.scale, qt.zero, qt.maxq,
        #    qt.sym, qt.r_bit, qt.r_group,
        #    qt.bits, qt.qrazor
        #).to(key_states.dtype).reshape(key_states_shape_)
    
        if self.config.k_pre_RoPE_quant:
            qt=Quantizer()
            qt.configure(
                self.config.bits_k, 
                self.config.k_qrazor_bits, 
                self.config.k_qrazor_group, 
                self.config.k_qrazor, 
                not self.config.k_perchannel, 
                self.config.sym_k, 
                mse=False
            )
            key_states_shape_ = key_states.shape
            key_states = key_states.reshape(-1, key_states_shape_[-1])
            qt.find_params(key_states, weight=True)
            key_states = quantize(
                key_states,
                qt.scale, qt.zero, qt.maxq,
                qt.sym, qt.r_bit, qt.r_group,
                qt.bits, qt.qrazor
            ).to(key_states.dtype).reshape(key_states_shape_)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Here we need to slice as we use a static cache by default, but FA2 does not support it
            if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                seq_len = attention_mask.shape[-1]
                key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

        attention_interface: Callable = gemma3_eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                    "Falling back to eager attention. This warning can be removed using the argument "
                    '`attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if attention_mask is not None:
            # backwards compatibility
            attention_mask = attention_mask.to(query_states)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

def qwen3_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        #qt=Quantizer()
        #qt.configure(
        #    bits=8, r_bit=4, r_group=32, qrazor=True, perchannel=True, sym=False, mse=False
        #)
        #key_states_shape_ = key_states.shape
        #key_states = key_states.reshape(-1, key_states_shape_[-1])
        #qt.find_params(key_states, weight=True)
        #key_states = quantize(
        #    key_states,
        #    qt.scale, qt.zero, qt.maxq,
        #    qt.sym, qt.r_bit, qt.r_group,
        #    qt.bits, qt.qrazor
        #).to(key_states.dtype).reshape(key_states_shape_)

        if self.config.k_pre_RoPE_quant:
            qt=Quantizer()
            qt.configure(
                self.config.bits_k, 
                self.config.k_qrazor_bits, 
                self.config.k_qrazor_group, 
                self.config.k_qrazor, 
                not self.config.k_perchannel, 
                self.config.sym_k, 
                mse=False
            )
            key_states_shape_ = key_states.shape
            key_states = key_states.reshape(-1, key_states_shape_[-1])
            qt.find_params(key_states, weight=True)
            key_states = quantize(
                key_states,
                qt.scale, qt.zero, qt.maxq,
                qt.sym, qt.r_bit, qt.r_group,
                qt.bits, qt.qrazor
            ).to(key_states.dtype).reshape(key_states_shape_)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = qwen3_eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights