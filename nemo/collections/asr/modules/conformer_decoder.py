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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.parts.utils import adapter_utils, rnnt_utils
from nemo.collections.common.parts import rnn
from nemo.core.classes import adapter_mixins, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AdapterModuleMixin
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, RelPositionalEncoding
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    ElementType,
    EmbeddedTextType,
    LabelsType,
    LengthsType,
    LogprobsType,
    LossType,
    NeuralType,
)
from nemo.utils import logging


class ConformerDecoder(rnnt_abstract.AbstractRNNTDecoder, Exportable, AdapterModuleMixin):
    """
    The decoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'abs_pos': absolute positional embedding and Transformer
            default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaults to 2048
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by sqrt(d_model)
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        conv_norm_type (str): the type of the normalization in the convolutional modules
            Defaults to 'batch_norm'.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
    """
    '''
    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(dev)
        input_example_length = torch.randint(1, max_dim, (max_batch,)).to(dev)
        return tuple([input_example, input_example_length])

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType(('B', 'H', 'T', 'D'), ElementType(), optional=True)],  # must always be last
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "prednet_lengths": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType((('B', 'H', 'T', 'D')), ElementType(), optional=True)],  # must always be last
        }
    '''
    def __init__(
        self,
        n_layers,
        d_model,
        vocab_size: int,
        feat_out=-1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=2048,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        blank_as_pad: bool = True,
        random_state_sampling: bool = False,
    ):        
        self.blank_idx = vocab_size
        
        # Initialize the model (blank token increases vocab size by 1)
        super().__init__(vocab_size=vocab_size, blank_idx=self.blank_idx, blank_as_pad=blank_as_pad)
        
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.scale = math.sqrt(self.d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.random_state_sampling = random_state_sampling

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        self._feat_out = d_model

        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        if self.blank_as_pad:
            self.embed = torch.nn.Embedding(vocab_size + 1, self.d_model, padding_idx=self.blank_idx)
        else:
            self.embed = torch.nn.Embedding(vocab_size, self.d_model)

        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        
        self.set_max_seq_length(self.pos_emb_max_len)
        
        self.use_pad_mask = True
        self._conft_export = False

    def set_max_seq_length(self, max_seq_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.
        """
        self.max_seq_length = max_seq_length
        device = next(self.parameters()).device
        seq_range = torch.arange(0, self.max_seq_length, device=device)
        if hasattr(self, 'seq_range'):
            self.seq_range = seq_range
        else:
            self.register_buffer('seq_range', seq_range, persistent=False)
        self.pos_enc.extend_pe(max_seq_length, device)
    
    def forward(self, targets, target_length, states=None):
        
        self.update_max_seq_length(seq_length=targets.size(-1), device=targets.device)
        y = rnn.label_collate(targets)
        
        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        if self._conft_export:
            add_sos = False
        elif self.training:
            add_sos = True
        else:
            add_sos = False
        
        g, states = self.predict(y, states, target_length, add_sos)
        g = g.transpose(1, 2)  # (B, D, U)
        
        return g, target_length, states

    def predict(
        self,
        y = None,
        state = None,
        length = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ):
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype
        
        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.embed(y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0][0].size(0)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.d_model), device=device, dtype=dtype)
        
        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        max_seq_length = y.size(1)

        if max_seq_length > self.max_seq_length:
            self.set_max_seq_length(max_seq_length)

        y, pos_emb = self.pos_enc(y)
        # adjust size
        max_seq_length = y.size(1)
        
        if length is None:
            length = y.new_full((y.size(0),), max_seq_length, dtype=torch.int32, device=self.seq_range.device)
        
        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(y)
            else:
                state = tuple([None] * len(self.layers))
        
        # Create the self-attention and padding masks
        pad_mask = self.make_pad_mask(max_seq_length, length)
        att_mask = pad_mask.unsqueeze(1).repeat([1, max_seq_length, 1])
        att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
        att_mask = att_mask.tril(diagonal=0)
        att_mask = ~att_mask

        if self.use_pad_mask:
            pad_mask = ~pad_mask
        else:
            pad_mask = None
            
        presents = ()
        for lth, (layer, layer_past) in enumerate(zip(self.layers, state)):
            y, state_out = layer(x=y, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask, layer_past=layer_past)
            presents = presents + (state_out,)

        if self.out_proj is not None:
            y = self.out_proj(y)
        
        return y, presents

    def update_max_seq_length(self, seq_length: int, device):
        # Find global max seq length across all nodes
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_seq_length:
            self.set_max_seq_length(seq_length)

    def make_pad_mask(self, max_seq_length, seq_lens):
        """Make masking for padding."""
        mask = self.seq_range[:max_seq_length].expand(seq_lens.size(0), -1) < seq_lens.unsqueeze(-1)
        return mask

    def enable_pad_mask(self, on=True):
        # On inference, user may chose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask
    
    def initialize_state(self, y):
        """
        Initialize the state of the RNN layers, with same dtype and device as input `y`.

        Args:
            y: A torch.Tensor whose device the generated states will be placed on.

        Returns:
            Tuple of torch.Tensor, each of shape [B, H, T, D], where
                L = Number of RNN layers
                B = Batch size
                H = Hidden size of RNN.
        """
        batch = y.size(0)
        if y.ndim == 1:
            T = 1
        else:
            T = y.size(1)
        if self.random_state_sampling and self.training:
            state = tuple(
                (torch.randn(batch, self.n_heads, T, self.d_head, dtype=y.dtype, device=y.device),
                torch.randn(batch, self.n_heads, T, self.d_head, dtype=y.dtype, device=y.device))
            for _ in range(len(self.layers)))

        else:
            state = tuple(
                (torch.zeros(batch, self.n_heads, T, self.d_head, dtype=y.dtype, device=y.device),
                torch.zeros(batch, self.n_heads, T, self.d_head, dtype=y.dtype, device=y.device))
            for _ in range(len(self.layers)))
        return state

    def score_hypothesis(self, hypothesis, cache):
        """
        Similar to the predict() method, instead this method scores a Hypothesis during beam search.
        Hypothesis is a dataclass representing one hypothesis in a Beam Search.

        Args:
            hypothesis: Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.

        Returns:
            Returns a tuple (y, states, lm_token) such that:
            y is a torch.Tensor of shape [1, 1, H] representing the score of the last token in the Hypothesis.
            state is a list of RNN states, each of shape [L, 1, H].
            lm_token is the final integer token of the hypothesis.
        """
        if hypothesis.dec_state is not None:
            device = hypothesis.dec_state[0][0].device
        else:
            _p = next(self.parameters())
            device = _p.device

        # parse "blank" tokens in hypothesis
        if len(hypothesis.y_sequence) > 0 and hypothesis.y_sequence[-1] == self.blank_idx:
            blank_state = True
        else:
            blank_state = False

        # Convert last token of hypothesis to torch.Tensor
        target = torch.full([1, 1], fill_value=hypothesis.y_sequence[-1], device=device, dtype=torch.long)
        lm_token = target[:, -1]  # [1]

        # Convert current hypothesis into a tuple to preserve in cache
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            # Obtain score for target token and new states
            if blank_state:
                y, new_state = self.predict(None, state=None, add_sos=False, batch_size=1)  # [1, 1, H]
            else:
                y, new_state = self.predict(target, state=hypothesis.dec_state, add_sos=False, batch_size=1)  # [1, 1, H]

            y = y[:, -1:, :]  # Extract just last state : [1, 1, H]
            #new_state = tuple((layer[0][:, :, -1:, :], layer[1][:, :, -1:, :]) for layer in new_state)
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token
    
    def batch_select_state(self, batch_states, idx: int):
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                ([L x (B, H)], [L x (B, H)])

            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ([L x (1, H)], [L x (1, H)])
        """
        if batch_states is not None:
            state_list = []
            for layer_id in range(len(batch_states)):
                states = tuple(t[idx, None] for t in batch_states[layer_id])
                state_list.append(states)

            return tuple(state_list)
        else:
            return None

    def batch_concat_states(self, batch_states):
        """Concatenate a batch of decoder state to a packed state.

        Args:
            batch_states (list): batch of decoder states
                B x ([L x (H)], [L x (H)])

        Returns:
            (tuple): decoder states
                (L x B x H, L x B x H)
        """
        print('batch_concat_states: ', type(batch_states), flush=True)
        print('batch_concat_states: ', len(batch_states), flush=True)
        
        if isinstance(batch_states, list):
            if len(batch_states) == 1:
                return batch_states[0]
            else:
                return self.batch_initialize_states(None, batch_states)
        else:
            return batch_states
    
    def batch_copy_states(
        self,
        old_states,
        new_states,
        ids,
        value = None,
    ):
        """Copy states from new state to old state at certain indices.

        Args:
            old_states(list): packed decoder states
                (L x B x H, L x B x H)

            new_states: packed decoder states
                (L x B x H, L x B x H)

            ids (list): List of indices to copy states at.

            value (optional float): If a value should be copied instead of a state slice, a float should be provided

        Returns:
            batch of decoder states with partial copy at ids (or a specific value).
                (L x B x H, L x B x H)
        """
        if len(ids) == 0:
            return old_states
        
        for layer_id in range(len(old_states)):
            if value is None:
                for idx in range(len(old_states[layer_id])):
                    #old_states[layer_id][idx][ids, :, :, :] = new_states[layer_id][idx][ids, :, :, :]
                    old_states[layer_id][idx][ids, :, -1, :] *= 0.0
            else:
                for idx in range(len(old_states[layer_id])):
                    old_states[layer_id][idx][ids, :, :, :] *= 0.0
                    old_states[layer_id][idx][ids, :, :, :] += value
        
        #x1 = tuple((layer[0][:, :, -1:, :], layer[1][:, :, -1:, :]) for layer in old_states)
        
        return old_states
    
    def batch_score_hypothesis(
        self, hypotheses: List[rnnt_utils.Hypothesis], cache, batch_states
    ):
        """
        Used for batched beam search algorithms. Similar to score_hypothesis method.

        Args:
            hypothesis: List of Hypotheses. Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.
            batch_states: List of torch.Tensor which represent the states of the RNN for this batch.
                Each state is of shape [L, B, H]

        Returns:
            Returns a tuple (b_y, b_states, lm_tokens) such that:
            b_y is a torch.Tensor of shape [B, 1, H] representing the scores of the last tokens in the Hypotheses.
            b_state is a list of list of RNN states, each of shape [L, B, H].
                Represented as B x List[states].
            lm_token is a list of the final integer tokens of the hypotheses in the batch.
        """
        final_batch = len(hypotheses)

        if final_batch == 0:
            raise ValueError("No hypotheses was provided for the batch!")

        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        # For each hypothesis, cache the last token of the sequence and the current states
        for i, hyp in enumerate(hypotheses):
            sequence = tuple(hyp.y_sequence)

            if sequence in cache:
                done[i] = cache[sequence]
            else:
                tokens.append(hyp.y_sequence[-1])
                process.append((sequence, hyp.dec_state))

        if process:
            batch = len(process)

            # convert list of tokens to torch.Tensor, then reshape.
            tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(batch, -1)
            #dec_states = self.initialize_state(tokens.to(dtype=dtype))  # [L, B, H]
            dec_states = self.batch_initialize_states(None, [d_state for seq, d_state in process])

            y, dec_states = self.predict(
                tokens, state=dec_states, add_sos=False, batch_size=batch
            )  # [B, 1, H], List([L, 1, H])

            #dec_states = tuple(state.to(dtype=dtype) for state in dec_states)
            dec_states = tuple((layer[0][:, :, :-1, :].to(dtype=dtype), layer[1][:, :, :-1, :].to(dtype=dtype)) for layer in dec_states)

        # Update done states and cache shared by entire batch.
        j = 0
        for i in range(final_batch):
            if done[i] is None:
                # Select sample's state from the batch state list
                new_state = self.batch_select_state(dec_states, j)

                # Cache [1, H] scores of the current y_j, and its corresponding state
                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        # Set the incoming batch states with the new states obtained from `done`.
        batch_states = self.batch_initialize_states(batch_states, [d_state for y_j, d_state in done])

        # Create batch of all output scores
        # List[1, 1, H] -> [B, 1, H]
        batch_y = torch.stack([y_j for y_j, d_state in done], dim=0)

        # Extract the last tokens from all hypotheses and convert to a tensor
        lm_tokens = torch.tensor([h.y_sequence[-1] for h in hypotheses], device=device, dtype=torch.long).view(final_batch)

        return batch_y, batch_states, lm_tokens
    
    def batch_initialize_states(self, batch_states, decoder_states):
        """
        Create batch of decoder states.

        Args:
           batch_states (list): batch of decoder states
              ([L x (B, H)], [L x (B, H)])

           decoder_states (list of list): list of decoder states
               [B x ([L x (1, H)], [L x (1, H)])]

        Returns:
           batch_states (tuple): batch of decoder states
               ([L x (B, H)], [L x (B, H)])
        """
        new_states = []
        for layer in range(len(self.layers)):
            k = torch.cat([x[layer][0] for x in decoder_states], dim=0)
            v = torch.cat([x[layer][1] for x in decoder_states], dim=0)
            new_states.append((k, v))

        return tuple(new_states)


# Add the adapter compatible modules to the registry
for cls in [ConformerDecoder]:
    if adapter_mixins.get_registered_adapter(cls) is None:
        adapter_mixins.register_adapter(cls, cls)  # base class is adapter compatible itself
