# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

import math

import numpy as np
import torch
from torch import nn
from torch.nn.functional import gelu

from einops import rearrange
from typing import Optional

from nemo.collections.common.parts import form_attention_mask
from nemo.utils import logging

__all__ = ["TransformerEmbedding", "AttentionBridge"]


class FixedPositionalEncoding(nn.Module):
    """
    Fixed positional encoding (embedding layer) from sine and cosine functions
    of different frequencies according to https://arxiv.org/abs/1706.03762

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
    """

    def __init__(self, hidden_size, max_sequence_length=512):
        super().__init__()

        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length
        self._build_pos_enc(hidden_size=self._hidden_size, max_sequence_length=self._max_sequence_length)

    def _build_pos_enc(self, hidden_size, max_sequence_length, device=None):
        """
        Builds/replaces pre-computed positional encoding.
        """
        pos_enc = torch.zeros(max_sequence_length, hidden_size, device=device)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        coef = -math.log(10000.0) / hidden_size
        div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.div_(math.sqrt(hidden_size))
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, position_ids):
        max_pos_id = position_ids.max()
        # update positional encoding if needed
        if max_pos_id >= self._max_sequence_length:
            logging.warning(
                f'Max position id {max_pos_id} is greater than max sequence length {self._max_sequence_length}. Expanding position embeddings just for this batch. This is not expected to work very well. Consider chunking your input into smaller sequences.'
            )
            self._build_pos_enc(
                hidden_size=self._hidden_size, max_sequence_length=max_pos_id + 1, device=position_ids.device,
            )

        embeddings = torch.embedding(self.pos_enc, position_ids)

        # Revert expansion of position embeddings since this wall checkpoint size mismatches.
        if max_pos_id >= self._max_sequence_length:
            self._build_pos_enc(
                hidden_size=self._hidden_size,
                max_sequence_length=self._max_sequence_length,
                device=position_ids.device,
            )
        return embeddings


class TransformerEmbedding(nn.Module):
    """
    Embedding from token and position embeddings.
    Optionally add token_type embedding (e.g. type of the sentence in BERT).

    Args:
        vocab_size: size of the vocabulary
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
        num_token_types: number of different token types
            (e.g. tokens of sentence A and tokens of sentence B in BERT)
        embedding_dropout: probability of dropout applied to embeddings
        learn_positional_encodings: whether to learn positional encodings or
            use fixed (sine-cosine) ones
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        padding_idx: int = 0,
        alibi: bool = False,
        embeddings_ln: bool = False,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.learn_positional_encodings = learn_positional_encodings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.alibi = alibi
        if learn_positional_encodings:
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        else:
            self.position_embedding = FixedPositionalEncoding(hidden_size, max_sequence_length)
        if num_token_types > 0:
            self.token_type_embedding = nn.Embedding(num_token_types, hidden_size)
        if embeddings_ln:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, input_ids, token_type_ids=None, start_pos=0, attention_mask=None):
        token_embeddings = self.token_embedding(input_ids)
        
        if not self.alibi:
            seq_length = input_ids.size(1)
            # we fail here only with parametric positional embedding. FixedPositionalEncoding automatically extends.
            if self.learn_positional_encodings and (seq_length > self.max_sequence_length):
                raise ValueError(
                    f"Input sequence is longer than maximum allowed sequence length for positional encoding. "
                    f"Got {seq_length} and {self.max_sequence_length}"
                )
            position_ids = torch.arange(
                start=start_pos, end=start_pos + seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)
            if attention_mask is not None:
                position_ids = torch.clamp(position_ids - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[:, start_pos:], min=0)
            position_embeddings = self.position_embedding(position_ids)
            
            embeddings = token_embeddings + position_embeddings
        else:
            embeddings = token_embeddings

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention layer.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
    """

    def __init__(self, hidden_size, num_attention_heads, attn_score_dropout=0.0, attn_layer_dropout=0.0):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number "
                "of attention heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_head_size = int(hidden_size / num_attention_heads)
        self.attn_scale = math.sqrt(math.sqrt(self.attn_head_size))

        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values, attention_mask):

        # attention_mask is needed to hide the tokens which correspond to [PAD]
        # in the case of BERT, or to hide the future tokens in the case of
        # vanilla language modeling and translation
        query = self.query_net(queries)
        key = self.key_net(keys)
        value = self.value_net(values)
        query = self.transpose_for_scores(query) / self.attn_scale
        key = self.transpose_for_scores(key) / self.attn_scale
        value = self.transpose_for_scores(value)

        # for numerical stability we pre-divide query and key by sqrt(sqrt(d))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        # output projection
        output_states = self.out_projection(context)
        output_states = self.layer_dropout(output_states)
        return output_states


class PositionWiseFF(nn.Module):
    """
    Position-wise feed-forward network of Transformer block.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        ffn_dropout: probability of dropout applied to net output
        hidden_act: activation function used between two linear layers
    """

    def __init__(self, hidden_size, inner_size, ffn_dropout=0.0, hidden_act="relu"):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, hidden_size)
        self.layer_dropout = nn.Dropout(ffn_dropout)
        ACT2FN = {"gelu": gelu, "relu": torch.relu}
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        output_states = self.dense_in(hidden_states)
        output_states = self.act_fn(output_states)
        output_states = self.dense_out(output_states)
        output_states = self.layer_dropout(output_states)
        return output_states


class AttentionBridge(torch.nn.Module):
    """
    A multi-head attention bridge to project a variable-size hidden states
    to k hidden states (per attention head).

    Code is based on the paper https://arxiv.org/pdf/1703.03130.pdf
    """

    def __init__(self, hidden_size, k, bridge_size):
        """
        hidden_size - size of input hidden state
        k - number of attention heads
        bridge_size - size of internal feed forward weights (i.e., attention head size)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.k = k
        self.bridge_size = bridge_size

        self.attn_scale = np.sqrt(np.sqrt(self.bridge_size))

        # build model

        self.W1 = torch.nn.Linear(hidden_size, bridge_size, bias=False)
        self.W2 = torch.nn.Linear(bridge_size, k, bias=False)
        self.act = torch.nn.ReLU()

    def forward(self, hidden, hidden_mask=None, return_ortho_loss=False):
        """
        Project hidden [B x N x H] to fixed-size [B x k x H]

        return_ortho_loss - if True returns loss term to encourage
                              orthogonal attention vectors
        """

        attention_scores = self.W2(self.act(self.W1(hidden) / self.attn_scale) / self.attn_scale).transpose(-1, -2)

        attention_mask = form_attention_mask(hidden_mask)
        if attention_mask is not None:
            attention_mask.squeeze_(1)
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)

        A = torch.softmax(attention_scores, dim=-1)
        M = A @ hidden

        if return_ortho_loss:
            ortho_loss = ((A @ A.transpose(-1, -2)) - torch.eye(self.k).type_as(A)).pow(2).sum()

            return M, ortho_loss
        else:
            return M


def _reset_is_causal(num_query_tokens: int, num_key_tokens: int, original_is_causal: bool):
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError('GPT does not support query and key with different number of tokens, unless number of query tokens is 1.')
        else:
            return False
    return original_is_causal


def scaled_multihead_dot_product_attention(query, key, value, n_heads, softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=1 if multiquery else n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=1 if multiquery else n_heads)
    
    min_val = torch.finfo(q.dtype).min
    (b, _, s_q, d) = q.shape
    s_k = k.size(-1)
    
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)
    
    attn_weight = q.matmul(k) * softmax_scale
    
    if attn_bias is not None:
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
            raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias
    
    if key_padding_mask is not None:
        if attn_bias is not None:
            logging.warning('Propogating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unneccessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
    
    if is_causal:
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)
    
    out = attn_weight.matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    
    if needs_weights:
        return (out, attn_weight)
    
    return (out, None)


def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r}).')


def flash_attn_fn(query, key, value, n_heads, softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False):
    try:
        from flash_attn import bert_padding, flash_attn_interface
    except:
        raise RuntimeError('Please install flash-attn==1.0.3.post0')
    
    check_valid_inputs(query, key, value)
    
    if attn_bias is not None:
        raise NotImplementedError('attn_bias not implemented for flash attn.')
    
    (batch_size, seqlen) = query.shape[:2]
    
    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
    
    query_padding_mask = key_padding_mask[:, -query.size(1):]
    (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding.unpad_input(query, query_padding_mask)
    query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)
    (key_unpad, _, cu_seqlens_k, max_seqlen_k) = bert_padding.unpad_input(key, key_padding_mask)
    key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=1 if multiquery else n_heads)
    (value_unpad, _, _, _) = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=1 if multiquery else n_heads)
    
    if multiquery:
        key_unpad = key_unpad.expand(key_unpad.size(0), n_heads, key_unpad.size(-1))
        value_unpad = value_unpad.expand(value_unpad.size(0), n_heads, value_unpad.size(-1))
    
    dropout_p = dropout_p if training else 0.0
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    output_unpad = flash_attn_interface.flash_attn_unpadded_func(query_unpad, key_unpad, value_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale=softmax_scale, causal=reset_is_causal, return_attn_probs=needs_weights)
    output = bert_padding.pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size, seqlen)
    
    return (output, None)


def triton_flash_attn_fn(query, key, value, n_heads, softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False):
    try:
        from flash_attn import flash_attn_triton
    except:
        raise RuntimeError('Please install flash-attn==1.0.3.post0 and triton==2.0.0.dev20221202')
    
    check_valid_inputs(query, key, value)
    
    if dropout_p:
        raise NotImplementedError('Dropout not implemented for attn_impl: triton.')
    
    if needs_weights:
        raise NotImplementedError('attn_impl: triton cannot return attn weights.')
    
    if key_padding_mask is not None:
        logging.warning('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        (b_size, s_k) = key_padding_mask.shape[:2]
        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)
        attn_bias = attn_bias.masked_fill(~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min)
    
    query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    key = rearrange(key, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads)
    value = rearrange(value, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads)
    
    if multiquery:
        key = key.expand(*key.shape[:2], n_heads, key.size(-1))
        value = value.expand(*value.shape[:2], n_heads, value.size(-1))
    
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_triton.flash_attn_func(query, key, value, attn_bias, reset_is_causal, softmax_scale)
    output = attn_output.view(*attn_output.shape[:2], -1)
    
    return (output, None)


class MultiheadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 attn_impl: str='torch',
                 clip_qkv: Optional[float]=None,
                 qk_ln: bool=False,
                 softmax_scale: Optional[float]=None,
                 attn_pdrop: float=0.0,
                 low_precision_layernorm: bool=False):
        
        super().__init__()
        
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        
        self.attn_dropout_p = attn_pdrop
        
        self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model)
        
        fuse_splits = (d_model, 2 * d_model)
        self.Wqkv._fused = (0, fuse_splits)
        
        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(self.d_model)
            self.k_ln = layernorm_class(self.d_model)
        
        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            #logging.warning('While `attn_impl: triton` can be faster than `attn_impl: flash` ' + 'it uses more memory. When training larger models this can trigger ' + 'alloc retries which hurts performance. If encountered, we recommend ' + 'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.')
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            #if torch.cuda.is_available():
            #    logging.warning('Using `attn_impl: torch`. If your model does not use `alibi` or ' + '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' + 'we recommend using `attn_impl: triton`.')
        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj._is_residual = True

    def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
        qkv = self.Wqkv(x)
        
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        
        (query, key, value) = qkv.chunk(3, dim=2)
        
        key_padding_mask = attention_mask
        
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        
        if past_key_value is not None:
            if len(past_key_value) != 0:
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)
            past_key_value = (key, value)
        
        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]
        
        (context, attn_weights) = self.attn_fn(query, key, value, self.n_heads, softmax_scale=self.softmax_scale, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights)
        
        return (self.out_proj(context), attn_weights, past_key_value)


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor

class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)

def rms_norm(x, weight=None, eps=1e-05):
    output = x / torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output

class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)

class LPRMSNorm(RMSNorm):

    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, weight=weight, dtype=dtype, device=device)

    def forward(self, x):
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight, self.eps).to(dtype=x.dtype)

NORM_CLASS_REGISTRY = {'layernorm': torch.nn.LayerNorm, 'low_precision_layernorm': LPLayerNorm, 'rmsnorm': RMSNorm, 'low_precision_rmsnorm': LPRMSNorm}
ATTN_CLASS_REGISTRY = {'multihead_attention': MultiheadAttention, 'multiquery_attention': None}
