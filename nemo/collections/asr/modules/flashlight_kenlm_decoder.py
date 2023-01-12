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
import torch
import itertools as it
import numpy as np

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import LengthsType, LogprobsType, NeuralType, PredictionsType


class FlashLightKenLMBeamSearchDecoder(NeuralModule):
    '''
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"hypos": NeuralType(('B'), PredictionsType())}
    '''
    
    class TokensWrapper:
        def __init__(self, asr_model):
            self.model = asr_model
            
            if not hasattr(asr_model, 'tokenizer'):
                self.reverse_map = {v:k for k,v in asr_model._wer.decoding.labels_map.items()}
        
        @property
        def blank(self):
            return self.model._wer.decoding.blank_id
        
        @property
        def unk_id(self):
            if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'unk_id'):
                return self.model.tokenizer.unk_id
            
            if '<unk>' in self.vocab:
                return self.token_to_id('<unk>')
            else:
                return -1
        
        @property
        def vocab(self):
            if hasattr(self.model, 'tokenizer'):
                return self.model.tokenizer.vocab
            else:
                return list(self.reverse_map.keys())
        
        @property
        def vocab_size(self):
            return self.model.decoder.num_classes_with_blank
        
        def token_to_id(self, token):
            if token == self.blank:
                return -1
            
            if hasattr(self.model, 'tokenizer'):
                return self.model.tokenizer.token_to_id(token)
            else:
                return self.reverse_map[token]
    
    
    def __init__(
        self,
        asr_model,
        lm_path,
        lexicon_path=None,
        nbest=1,
        beam_size=32,
        beam_size_token=32,
        beam_threshold=25.0,
        lm_weight=2.0,
        word_score=-1.0,
        unk_weight=-math.inf,
        sil_weight=0.0,
        unit_lm=False,
    ):

        try:
            from flashlight.lib.text.dictionary import create_word_dict, load_words
            from flashlight.lib.sequence.criterion import get_data_ptr_as_bytes
            from flashlight.lib.text.decoder import (
                CriterionType,
                LexiconDecoderOptions,
                KenLM,
                LM,
                LMState,
                SmearingMode,
                Trie,
                LexiconDecoder,
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "FlashLightKenLMBeamSearchDecoder requires the installation of flashlight python bindings"
            )

        super().__init__()
        
        self.criterion_type = CriterionType.CTC
        self.nbest = nbest
        self.tokenizer = self.TokensWrapper(asr_model)
        #self.vocab_size = tokenizer.vocab_size + 1
        #self.blank = int(tokenizer.vocab_size)
        self.vocab_size = self.tokenizer.vocab_size
        self.blank = self.tokenizer.blank
        self.silence = self.tokenizer.unk_id
        self.unit_lm = unit_lm
            
        if lexicon_path is not None:
            self.lexicon = load_words(lexicon_path)
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")
    
            self.lm = KenLM(lm_path, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)
    
            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    spelling_idxs = [self.tokenizer.token_to_id(token) for token in spelling]
                    #assert (self.tokenizer.unk_id not in spelling_idxs), f"{spelling} {spelling_idxs}"
                    if self.tokenizer.unk_id in spelling_idxs:
                        print(f'tokenizer has unknown id for word[ {word} ] {spelling} {spelling_idxs}', flush=True)
                        continue
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)
    
            self.decoder_opts = LexiconDecoderOptions(
                beam_size=beam_size,
                beam_size_token=int(beam_size_token),
                beam_threshold=beam_threshold,
                lm_weight=lm_weight,
                word_score=word_score,
                unk_score=unk_weight,
                sil_score=sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            
            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                [],
                self.unit_lm,
            )
        else:
            assert unit_lm, "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

            d = {w: [[w]] for w in self.tokenizer.vocab}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(lm_path, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=beam_size,
                beam_size_token=int(beam_size_token),
                beam_threshold=beam_threshold,
                lm_weight=lm_weight,
                sil_score=sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )
    
    def _get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))
    
    def _get_timesteps(self, token_idxs):
        """Returns frame numbers corresponding to every non-blank token.
        Parameters
        ----------
        token_idxs : List[int]
            IDs of decoded tokens.
        Returns
        -------
        List[int]
            Frame numbers corresponding to every non-blank token.
        """
        timesteps = []
        for i, token_idx in enumerate(token_idxs):
            if token_idx == self.blank:
                continue
            if i == 0 or token_idx != token_idxs[i-1]:
                timesteps.append(i)
        return timesteps

    #@typecheck(ignore_collections=True)
    @torch.no_grad()
    def forward(self, log_probs):
        if type(log_probs) is not torch.Tensor:
            log_probs = torch.from_numpy(log_probs).float()
        if log_probs.dim() == 2:
            log_probs = log_probs.unsqueeze(0)
        
        emissions = log_probs.cpu().contiguous()
        
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self._get_tokens(result.tokens),
                        "score": result.score,
                        "timesteps": self._get_timesteps(result.tokens),
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    }
                    for result in nbest_results
                ]
            )
        
        return hypos
        