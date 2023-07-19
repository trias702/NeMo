# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


"""
A script to convert a Nemo ASR Hybrid model file (.nemo) to a Nemo ASR CTC model file (.nemo)

This allows you to train a RNNT-CTC Hybrid model, but then convert to a pure CTC model for use
in Riva. Works just fine with nemo2riva, HOWEVER, Riva doesn't support AggTokenizer, but nemo2riva
does, so be careful that you do not convert a model with AggTokenizer and then use that in Riva
as it will not work.

Usage: python convert_nemo_asr_hybrid_to_ctc.py -i /path/to/hybrid.nemo -o /path/to/saved_ctc_model.nemo

"""


import argparse
import os
import torch
from nemo.collections.asr.models import ASRModel, EncDecCTCModel, EncDecCTCModelBPE

from omegaconf import OmegaConf

from nemo.utils import logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', required=True, type=str, help='path to Nemo Hybrid model .nemo file'
    )
    parser.add_argument(
        '-o', '--output', required=True, type=str, help='path and name of output .nemo file'
    )
    parser.add_argument('--cuda', action='store_true', help='put Nemo model onto GPU prior to savedown')

    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logging.critical(f'Input file [ {args.input} ] does not exist or cannot be found. Aborting.')
        exit(255)
    
    hybrid_model = ASRModel.restore_from(args.input, map_location=torch.device('cpu'))
    
    BPE = False
    ctc_class = EncDecCTCModel
    if 'tokenizer' in hybrid_model.cfg.keys():
        BPE = True
        ctc_class = EncDecCTCModelBPE
    
    hybrid_model_cfg = OmegaConf.to_container(hybrid_model.cfg)
    
    new_cfg = {}
    new_cfg['sample_rate'] = hybrid_model_cfg['sample_rate']
    new_cfg['log_prediction'] = hybrid_model_cfg['log_prediction']
    new_cfg['ctc_reduction'] = hybrid_model_cfg['aux_ctc']['ctc_reduction']
    new_cfg['skip_nan_grad'] = hybrid_model_cfg['skip_nan_grad']
    if BPE:
        new_cfg['tokenizer'] = hybrid_model_cfg['tokenizer']
    new_cfg['preprocessor'] = hybrid_model_cfg['preprocessor']
    new_cfg['spec_augment'] = hybrid_model_cfg['spec_augment']
    new_cfg['encoder'] = hybrid_model_cfg['encoder']
    new_cfg['decoder'] = hybrid_model_cfg['aux_ctc']['decoder']
    new_cfg['interctc'] = hybrid_model_cfg['interctc']
    new_cfg['optim'] = hybrid_model_cfg['optim']
    new_cfg['train_ds'] = hybrid_model_cfg['train_ds']
    new_cfg['validation_ds'] = hybrid_model_cfg['validation_ds']
    
    new_cfg_oc = OmegaConf.create(new_cfg)
    
    ctc_model = ctc_class.restore_from(args.input, map_location=torch.device('cpu'), override_config_path=new_cfg_oc, strict=False)
    
    assert all([torch.allclose(hybrid_model.state_dict()[x], ctc_model.state_dict()[x]) for x in hybrid_model.state_dict().keys() if x.split('.')[0] in ['preprocessor', 'encoder']]), "Encoder and preprocessor state dicts don't match!"
    
    ctc_model.decoder.load_state_dict(hybrid_model.ctc_decoder.state_dict())
    
    assert all([torch.allclose(hybrid_model.ctc_decoder.state_dict()[x], ctc_model.decoder.state_dict()[x]) for x in hybrid_model.ctc_decoder.state_dict().keys()]), "Decoder state_dict load failed!"
    
    if args.cuda and torch.cuda.is_available():
        ctc_model = ctc_model.cuda()
    
    ctc_model.save_to(args.output)
    logging.info(f'Converted CTC model was successfully saved to {args.output}')
