# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import torch
import torch.nn as nn

from typing import Optional

from nemo.collections.common.parts import form_attention_mask, attn_bias_shape, build_attn_bias
from nemo.collections.nlp.modules.common.transformer.transformer_modules import MultiHeadAttention, PositionWiseFF, NORM_CLASS_REGISTRY, ATTN_CLASS_REGISTRY

__all__ = ["TransformerEncoder"]


'''
class TransformerEncoderBlock(nn.Module):
    """
    Building block of Transformer encoder.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            attention layers, but before layer normalization
        ffn_dropout: probability of dropout applied to FFN output
        hidden_act: activation function used between two linear layers in FFN
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.second_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)

    def forward_preln(self, encoder_query, encoder_mask, encoder_keys):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = encoder_query
        encoder_query = self.layer_norm_1(encoder_query)
        encoder_keys = self.layer_norm_1(encoder_keys)
        self_attn_output = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        output_states = self.second_sub_layer(self_attn_output)
        output_states += residual

        return output_states

    def forward_postln(self, encoder_query, encoder_mask, encoder_keys):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        self_attn_output = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += encoder_query
        self_attn_output = self.layer_norm_1(self_attn_output)

        output_states = self.second_sub_layer(self_attn_output)
        output_states += self_attn_output
        output_states = self.layer_norm_2(output_states)

        return output_states

    def forward(self, encoder_query, encoder_mask, encoder_keys):
        if self.pre_ln:
            return self.forward_preln(encoder_query, encoder_mask, encoder_keys)
        else:
            return self.forward_postln(encoder_query, encoder_mask, encoder_keys)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        mask_future: bool = False,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
    ):
        super().__init__()

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        layer = TransformerEncoderBlock(
            hidden_size,
            inner_size,
            num_attention_heads,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            hidden_act,
            pre_ln,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.diag = 0 if mask_future else None

    def _get_memory_states(self, encoder_states, encoder_mems_list=None, i=0):
        if encoder_mems_list is not None:
            memory_states = torch.cat((encoder_mems_list[i], encoder_states), dim=1)
        else:
            memory_states = encoder_states
        return memory_states

    def forward(self, encoder_states, encoder_mask, encoder_mems_list=None, return_mems=False):
        """
        Args:
            encoder_states: output of the embedding_layer (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            encoder_mems_list: list of the cached encoder hidden states
                for fast autoregressive generation which will be used instead
                of encoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all encoder layers
                or the last layer only
        """

        encoder_attn_mask = form_attention_mask(encoder_mask, self.diag)

        memory_states = self._get_memory_states(encoder_states, encoder_mems_list, 0)
        cached_mems_list = [memory_states]

        for i, layer in enumerate(self.layers):
            encoder_states = layer(encoder_states, encoder_attn_mask, memory_states)
            memory_states = self._get_memory_states(encoder_states, encoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if self.final_layer_norm is not None:
            encoder_states = self.final_layer_norm(encoder_states)
            memory_states = self._get_memory_states(encoder_states, encoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if return_mems:
            return cached_mems_list
        else:
            return cached_mems_list[-1]
'''


class MPTMLP(nn.Module):
    
    def __init__(self, d_model: int, expansion_ratio: int):
        super().__init__()
        
        self.up_proj = nn.Linear(d_model, expansion_ratio * d_model)
        self.act = nn.GELU(approximate='none')
        self.down_proj = nn.Linear(expansion_ratio * d_model, d_model)
        self.down_proj._is_residual = True

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class TransformerEncoderBlock(nn.Module):
    """
    Building block of Transformer encoder.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            attention layers, but before layer normalization
        ffn_dropout: probability of dropout applied to FFN output
        hidden_act: activation function used between two linear layers in FFN
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 1,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        expansion_ratio: int = 4,
        norm_type: str = "low_precision_layernorm",
        attn_type: str = "multihead_attention",
        attn_impl: str = "torch",
        pre_ln: bool = True,
        qk_ln: bool = False,
        is_causal: bool = True,
        clip_qkv: Optional[float] = None,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__()
        
        self.pre_ln = pre_ln
        self.is_causal = is_causal
        
        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        attn_class = ATTN_CLASS_REGISTRY[attn_type.lower()]
        self.norm_1 = norm_class(hidden_size)
        self.attn = attn_class(attn_impl=attn_impl, clip_qkv=clip_qkv, qk_ln=qk_ln, softmax_scale=softmax_scale, attn_pdrop=attn_layer_dropout, d_model=hidden_size, n_heads=num_attention_heads)
        self.norm_2 = norm_class(hidden_size)
        self.ffn = MPTMLP(d_model=hidden_size, expansion_ratio=expansion_ratio)
        self.resid_attn_dropout = nn.Dropout(ffn_dropout)
        self.resid_ffn_dropout = nn.Dropout(ffn_dropout)
        

    def forward_preln(self, x, past_key_value=None, attn_bias=None, attention_mask=None):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        a = self.norm_1(x)
        
        (b, _, past_key_value) = self.attn(a, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=self.is_causal)
        
        x = x + self.resid_attn_dropout(b)
        m = self.norm_2(x)
        n = self.ffn(m)
        x = x + self.resid_ffn_dropout(n)
        
        return (x, past_key_value)

    def forward_postln(self, x, past_key_value=None, attn_bias=None, attention_mask=None):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """        
        (a, _, past_key_value) = self.attn(x, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=self.is_causal)
        
        x = x + self.resid_attn_dropout(a)
        m = self.norm_1(x)
        n = self.ffn(m)
        x = x + self.resid_ffn_dropout(n)
        x = self.norm_2(x)

        return (x, past_key_value)

    def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None):
        if self.pre_ln:
            return self.forward_preln(x=x, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask)
        else:
            return self.forward_postln(x=x, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int = 1,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        expansion_ratio: int = 4,
        max_sequence_length: int = 512,
        alibi_bias_max: int = 8,
        hidden_act: str = "gelu",
        norm_type: str = "low_precision_layernorm",
        attn_type: str = "multihead_attention",
        attn_impl: str = "torch",
        pre_ln: bool = True,
        pre_ln_final_layer_norm: bool = True,
        qk_ln: bool = False,
        prefix_lm: bool = False,
        use_alibi: bool = False,
        use_sequence_id: bool = False,
        clip_qkv: Optional[float] = None,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.prefix_lm = prefix_lm
        self.is_causal = not prefix_lm
        self.max_sequence_length = max_sequence_length
        self.use_alibi = use_alibi
        self.use_sequence_id = use_sequence_id
        self.alibi_bias_max = alibi_bias_max
        self._attn_impl = attn_impl
        self._attn_bias_initialized = False
        self.attn_bias = None
        self._attn_bias_shape = attn_bias_shape(attn_impl, num_attention_heads, max_sequence_length, use_alibi, prefix_lm=self.prefix_lm, causal=self.is_causal, use_sequence_id=self.use_sequence_id)
        
        if norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = ' | '.join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(f'Requested norm type ({norm_type}) is not implemented within this repo (Options: {norm_options}).')

        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]

        self.blocks = nn.ModuleList([TransformerEncoderBlock(hidden_size,
                                                             num_attention_heads,
                                                             attn_layer_dropout,
                                                             ffn_dropout,
                                                             expansion_ratio,
                                                             norm_type,
                                                             attn_type,
                                                             attn_impl,
                                                             pre_ln,
                                                             qk_ln,
                                                             self.is_causal,
                                                             clip_qkv,
                                                             softmax_scale) for _ in range(num_layers)])
        
        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = norm_class(self.hidden_size)
        else:
            self.final_layer_norm = None

    @torch.no_grad()
    def _attn_bias(self, device, dtype, attention_mask=None, prefix_mask=None, sequence_id=None):
        if not self._attn_bias_initialized:
            if self._attn_bias_shape:
                self.attn_bias = torch.zeros(self._attn_bias_shape, device=device, dtype=dtype)
                self.attn_bias = build_attn_bias(self._attn_impl, self.attn_bias, self.num_attention_heads, self.max_sequence_length, causal=self.is_causal, alibi=self.use_alibi, alibi_bias_max=self.alibi_bias_max)
            self._attn_bias_initialized = True
        
        if self._attn_impl == 'flash':
            return (self.attn_bias, attention_mask)
        
        if self.attn_bias is not None:
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
        
        attn_bias = self.attn_bias
        
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)
            assert isinstance(prefix_mask, torch.Tensor)
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
        
        if self.use_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)
            attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
        
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                attn_bias = attn_bias[:, :, :, -s_k:]
            if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
                raise ValueError(f'attention_mask shape={attention_mask.shape} ' + f'and prefix_mask shape={prefix_mask.shape} are not equal.')
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)
        
        return (attn_bias, None)

    def _apply_prefix_mask(self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor):
        (s_k, s_q) = attn_bias.shape[-2:]
        
        if s_k != self.max_sequence_length or s_q != self.max_sequence_length:
            raise ValueError('attn_bias does not match the expected shape. ' + f'The last two dimensions should both be {self.max_sequence_length} ' + f'but are {s_k} and {s_q}.')
        
        seq_len = prefix_mask.shape[-1]
        
        if seq_len > self.max_sequence_length:
            raise ValueError(f'prefix_mask sequence length cannot exceed max_seq_len={self.max_sequence_length}')
        
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        
        return attn_bias

    def _apply_sequence_id(self, attn_bias: torch.Tensor, sequence_id: torch.LongTensor):
        seq_len = sequence_id.shape[-1]
        
        if seq_len > self.max_sequence_length:
            raise ValueError(f'sequence_id sequence length cannot exceed max_seq_len={self.max_sequence_length}')
        
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        cannot_attend = torch.logical_not(torch.eq(sequence_id.view(-1, seq_len, 1), sequence_id.view(-1, 1, seq_len))).unsqueeze(1)
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        
        return attn_bias

    def forward(self, x, attention_mask=None, past_key_values=None, prefix_mask=None, sequence_id=None, output_hidden_states=False, use_cache=False):
        """
        Args:
            encoder_states: output of the embedding_layer (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            encoder_mems_list: list of the cached encoder hidden states
                for fast autoregressive generation which will be used instead
                of encoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all encoder layers
                or the last layer only
        """

        #attention_mask = form_attention_mask(attention_mask, self.diag)
        
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()

        (attn_bias, attention_mask) = self._attn_bias(device=x.device, dtype=x.dtype, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id)
        
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.num_layers)]
        
        all_hidden_states = () if output_hidden_states else None
        
        for (b_idx, block) in enumerate(self.blocks):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (x,)
                
            past_key_value = past_key_values[b_idx] if past_key_values is not None else None
            
            (x, past_key_value) = block(x, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask)
            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value
        
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        
        return x, past_key_values, all_hidden_states
