import torch
from typing import List, Optional, Tuple, Union
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_func
import internvl.model
import transformers
import torch.distributed as dist
from internvl.model.internlm2.modeling_internlm2 import (
    INTERNLM2_ATTENTION_CLASSES, InternLM2FlashAttention2,
    apply_rotary_pos_emb)
from internvl.model.internlm2.configuration_internlm2 import InternLM2Config
from torch import nn
# from ring_flash_attn.zigzag_ring_flash_attn import zigzag_ring_flash_attn_func
from ring_flash_attn.zigzag_ring_flash_attn_varlen import zigzag_ring_flash_attn_varlen_func



# Modified from internvl.model.internlm2.modeling_internlm2.InternLM2FlashAttention2
class InternLM2FlashAttention2ForPackedTraining(InternLM2FlashAttention2):

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                rename from cu_seqlens to keep compatability - (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                    of the sequences in the batch.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        cu_seqlens = attention_mask.squeeze(0)
        with torch.no_grad():
            max_seqlen = max([
                cu_seqlens[idx+1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]).item()

        # Contains at least one padding token in the sequence
        causal = self.is_causal and query_length != 1
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        if torch.isnan(attn_output).any():
            print("causal", causal)
            input()
            raise ValueError('Attention output contains NaN values')
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)
        return attn_output
class InternLM2RingAttention2ForPackedTraining(InternLM2FlashAttention2):
    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None,group=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                rename from cu_seqlens to keep compatability - (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                    of the sequences in the batch.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        cu_seqlens = attention_mask.squeeze(0)
        with torch.no_grad():
            max_seqlen = max([
                cu_seqlens[idx+1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]).item()
        # Contains at least one padding token in the sequence
        causal = self.is_causal and query_length != 1
        attn_output = zigzag_ring_flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            group=group
        )
        if torch.isnan(attn_output).any():
            print("causal", causal)
            raise ValueError('Attention output contains NaN values')
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)
        return attn_output


def replace_internlm2_attention_class(attn_type='packed'):

    if attn_type=='packed':
        INTERNLM2_ATTENTION_CLASSES['flash_attention_2'] = InternLM2FlashAttention2ForPackedTraining
    elif attn_type=='ring':
        print("replacing to ring attn")
        INTERNLM2_ATTENTION_CLASSES['flash_attention_2'] = InternLM2RingAttention2ForPackedTraining
    else:
        raise NotImplementedError()
    print('Replace INTERNLM2_ATTENTION_CLASSES to support packed training!!')




