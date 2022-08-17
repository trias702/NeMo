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
#

# This script would evaluate an N-gram language model trained with KenLM library (https://github.com/kpu/kenlm) in
# fusion with beam search decoders on top of a trained ASR model. NeMo's beam search decoders are capable of using the
# KenLM's N-gram models to find the best candidates. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# You may train the LM model with 'scripts/ngram_lm/train_kenlm.py'.
#
# USAGE: python eval_beamsearch_ngram.py --nemo_model_file <path to the .nemo file of the model> \
#                                         --input_manifest <path to the evaluation JSON manifest file \
#                                         --kenlm_model_file <path to the binary KenLM model> \
#                                         --beam_width <list of the beam widths> \
#                                         --beam_alpha <list of the beam alphas> \
#                                         --beam_beta <list of the beam betas> \
#                                         --preds_output_folder <optional folder to store the predictions> \
#                                         --decoding_mode beamsearch_ngram
#                                         ...
#
# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html


# Please check train_kenlm.py to find out why we need TOKEN_OFFSET for BPE-based models
TOKEN_OFFSET = 100

import argparse
import contextlib
import json
import os
import pickle
from pathlib import Path

import editdistance
import kenlm_utils
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils import logging


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def beam_search_eval(
    all_probs,
    target_transcripts,
    vocab,
    ids_to_text_func=None,
    preds_output_file=None,
    lm_path=None,
    beam_alpha=1.0,
    beam_beta=0.0,
    beam_width=128,
    beam_batch_size=128,
    progress_bar=True,
):
    # creating the beam search decoder
    beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
        vocab=vocab,
        beam_width=beam_width,
        alpha=beam_alpha,
        beta=beam_beta,
        lm_path=lm_path,
        num_cpus=max(int(os.cpu_count() * 0.75), 1),
        input_tensor=False,
    )

    wer_dist_first = cer_dist_first = 0
    wer_dist_best = cer_dist_best = 0
    words_count = 0
    chars_count = 0
    sample_idx = 0

    if progress_bar:
        it = tqdm(
            range(int(np.ceil(len(all_probs) / beam_batch_size))),
            #desc=f"Beam search decoding with width={beam_width}, alpha={beam_alpha}, beta={beam_beta}",
            ncols=120,
        )
    else:
        it = range(int(np.ceil(len(all_probs) / beam_batch_size)))
    for batch_idx in it:
        # disabling type checking
        with nemo.core.typecheck.disable_checks():
            probs_batch = all_probs[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]
            #beams_batch = beam_search_lm.forward(log_probs=probs_batch, log_probs_length=None,)
            beams_batch = beam_search_lm.forward(log_probs=[kenlm_utils.softmax(logits) for logits in probs_batch], log_probs_length=None,)

        for beams_idx, beams in enumerate(beams_batch):
            target = target_transcripts[sample_idx + beams_idx]
            target_split_w = target.split()
            target_split_c = list(target)
            words_count += len(target_split_w)
            chars_count += len(target_split_c)
            wer_dist_min = cer_dist_min = 10000
            for candidate_idx, candidate in enumerate(beams):
                if ids_to_text_func is not None:
                    # For BPE encodings, need to shift by TOKEN_OFFSET to retrieve the original sub-word ids
                    pred_text = ids_to_text_func([ord(c) - TOKEN_OFFSET for c in candidate[1]])
                    #pred_text = candidate[1]
                else:
                    pred_text = candidate[1]
                pred_split_w = pred_text.split()
                wer_dist = editdistance.eval(target_split_w, pred_split_w)
                pred_split_c = list(pred_text)
                cer_dist = editdistance.eval(target_split_c, pred_split_c)

                wer_dist_min = min(wer_dist_min, wer_dist)
                cer_dist_min = min(cer_dist_min, cer_dist)

                if candidate_idx == 0:
                    # first candidate
                    wer_dist_first += wer_dist
                    cer_dist_first += cer_dist

                score = candidate[0]
                if preds_output_file:
                    preds_output_file.write('{}\t{}\n'.format(pred_text, score))
            wer_dist_best += wer_dist_min
            cer_dist_best += cer_dist_min
        sample_idx += len(probs_batch)

    return wer_dist_first, cer_dist_first, wer_dist_best, cer_dist_best, words_count, chars_count


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an ASR model with beam search decoding and n-gram KenLM language model.'
    )
    parser.add_argument(
        "--nemo_model_file",
        required=True,
        type=str,
        help="The path of the '.nemo' file of the ASR model or name of a pretrained model",
    )
    parser.add_argument(
        "--kenlm_model_file", required=False, default=None, type=str, help="The path of the KenLM binary model file"
    )
    parser.add_argument("--input_manifest", required=True, type=str, help="The manifest file of the evaluation set")
    parser.add_argument(
        "--preds_output_folder", default=None, type=str, help="The optional folder where the predictions are stored"
    )
    parser.add_argument(
        "--probs_cache_file", default=None, type=str, help="The cache file for storing the outputs of the model"
    )
    parser.add_argument(
        "--acoustic_batch_size", default=16, type=int, help="The batch size to calculate log probabilities"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="The device to load the model onto to calculate log probabilities"
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Whether to use AMP if available to calculate log probabilities"
    )
    parser.add_argument(
        "--decoding_mode",
        choices=["greedy", "beamsearch", "beamsearch_ngram"],
        default="beamsearch_ngram",
        type=str,
        help="The decoding scheme to be used for evaluation.",
    )
    parser.add_argument(
        "--beam_width",
        required=False,
        type=int,
        nargs="+",
        help="The width or list of the widths for the beam search decoding",
    )
    parser.add_argument(
        "--beam_alpha",
        required=False,
        type=float,
        nargs="+",
        help="The alpha parameter or list of the alphas for the beam search decoding",
    )
    parser.add_argument(
        "--beam_beta",
        required=False,
        type=float,
        nargs="+",
        help="The beta parameter or list of the betas for the beam search decoding",
    )
    parser.add_argument(
        "--beam_batch_size", default=32, type=int, help="The batch size to be used for beam search decoding"
    )
    args = parser.parse_args()

    if args.nemo_model_file.endswith('.nemo'):
        asr_model = nemo_asr.models.ASRModel.restore_from(args.nemo_model_file, map_location=torch.device(args.device))
    else:
        logging.warning(
            "nemo_model_file does not end with .nemo, therefore trying to load a pretrained model with this name."
        )
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            args.nemo_model_file, map_location=torch.device(args.device)
        )

    target_transcripts = []
    manifest_dir = Path(args.input_manifest).parent
    with open(args.input_manifest, 'r') as manifest_file:
        audio_file_paths = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {args.input_manifest} ...", ncols=120):
            data = json.loads(line)
            audio_file = Path(data['audio_filepath'])
            if not audio_file.is_file() and not audio_file.is_absolute():
                audio_file = manifest_dir / audio_file
            target_transcripts.append(data['text'])
            audio_file_paths.append(str(audio_file.absolute()))

    if args.use_amp:
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            logging.info("AMP is enabled!\n")
            autocast = torch.cuda.amp.autocast
    else:
        @contextlib.contextmanager
        def autocast():
            yield

    wer_dist_greedy = 0
    cer_dist_greedy = 0
    words_count_greedy = 0
    chars_count_greedy = 0
    
    #wer_dist_first = cer_dist_first = 0
    #wer_dist_best = cer_dist_best = 0
    #words_count_lm = 0
    #chars_count_lm = 0
    
    if args.decoding_mode in ["beamsearch_ngram", "beamsearch"]:
        if args.beam_width is None or args.beam_alpha is None or args.beam_beta is None:
            raise ValueError("beam_width, beam_alpha and beam_beta are needed to perform beam search decoding.")
        params = {'beam_width': args.beam_width, 'beam_alpha': args.beam_alpha, 'beam_beta': args.beam_beta}
        hp_grid = ParameterGrid(params)
        hp_grid = list(hp_grid)

        logging.info("==============================Starting the beam search decoding===============================")
        logging.info(f"Grid search size: {len(hp_grid)}")
        logging.info("It may take some time...")
        logging.info("==============================================================================================")
        
        if args.preds_output_folder and not os.path.exists(args.preds_output_folder):
            os.mkdir(args.preds_output_folder)
        
        hp_grid_results = {}
        for hp in hp_grid:
            hp_grid_results[tuple(hp.values())] = {'wer_dist_first': 0, 'cer_dist_first': 0, 'wer_dist_best': 0,
                                   'cer_dist_best': 0, 'words_count_lm': 0, 'chars_count_lm': 0}
            
            if args.preds_output_folder:
                preds_output_file = os.path.join(
                    args.preds_output_folder,
                    f"preds_out_width{hp['beam_width']}_alpha{hp['beam_alpha']}_beta{hp['beam_beta']}.tsv",
                )
            else:
                preds_output_file = None
            
            if preds_output_file:
                hp_grid_results[tuple(hp.values())]['out_file'] = open(preds_output_file, 'w', encoding='utf_8', newline='\n')
            else:
                hp_grid_results[tuple(hp.values())]['out_file'] = None
                
    
    for chunk in chunks(list(zip(audio_file_paths, target_transcripts)), 2000):
        ch_audios = [x[0] for x in chunk]
        ch_targets = [x[-1] for x in chunk]
        with autocast():
            with torch.no_grad():
                all_probs = asr_model.transcribe(ch_audios, batch_size=args.acoustic_batch_size, logprobs=True)
        #all_probs = [kenlm_utils.softmax(logits) for logits in all_logits]
    
        
        for batch_idx, probs in enumerate(all_probs):
            preds = np.argmax(probs, axis=1)
            preds_tensor = torch.tensor(preds, device='cpu').unsqueeze(0)
            pred_text = asr_model._wer.decoding.ctc_decoder_predictions_tensor(preds_tensor)[0][0]
    
            pred_split_w = pred_text.split()
            target_split_w = ch_targets[batch_idx].split()
            pred_split_c = list(pred_text)
            target_split_c = list(ch_targets[batch_idx])
    
            wer_dist = editdistance.eval(target_split_w, pred_split_w)
            cer_dist = editdistance.eval(target_split_c, pred_split_c)
    
            wer_dist_greedy += wer_dist
            cer_dist_greedy += cer_dist
            words_count_greedy += len(target_split_w)
            chars_count_greedy += len(target_split_c)
    
        #logging.info('Greedy WER/CER = {:.2%}/{:.2%}'.format(wer_dist_greedy / words_count, cer_dist_greedy / chars_count))
    
        encoding_level = kenlm_utils.SUPPORTED_MODELS.get(type(asr_model).__name__, None)
        if not encoding_level:
            logging.warning(
                f"Model type '{type(asr_model).__name__}' may not be supported. Would try to train a char-level LM."
            )
            encoding_level = 'char'
    
        vocab = asr_model.decoder.vocabulary
        ids_to_text_func = None
        if encoding_level == "subword":
            vocab = [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab))]
            ids_to_text_func = asr_model.tokenizer.ids_to_text
        # delete the model to free the memory
        #del asr_model
    
        if args.decoding_mode == "beamsearch_ngram":
            if not os.path.exists(args.kenlm_model_file):
                raise FileNotFoundError(f"Could not find the KenLM model file '{args.kenlm_model_file}'.")
            lm_path = args.kenlm_model_file
        else:
            lm_path = None
    
        # 'greedy' decoding_mode would skip the beam search decoding
        if args.decoding_mode in ["beamsearch_ngram", "beamsearch"]:
    
            for hp in hp_grid:
                res_dict = hp_grid_results[tuple(hp.values())]
    
                wer_dist_first_, cer_dist_first_, wer_dist_best_, cer_dist_best_, words_count_lm_, chars_count_lm_ = beam_search_eval(
                    all_probs=all_probs,
                    target_transcripts=ch_targets,
                    vocab=vocab,
                    ids_to_text_func=ids_to_text_func,
                    preds_output_file=res_dict['out_file'],
                    lm_path=lm_path,
                    beam_width=hp["beam_width"],
                    beam_alpha=hp["beam_alpha"],
                    beam_beta=hp["beam_beta"],
                    beam_batch_size=args.beam_batch_size,
                    progress_bar=True,
                )
                
                res_dict['wer_dist_first'] += wer_dist_first_
                res_dict['cer_dist_first'] += cer_dist_first_
                res_dict['wer_dist_best'] += wer_dist_best_
                res_dict['cer_dist_best'] += cer_dist_best_
                res_dict['words_count_lm'] += words_count_lm_
                res_dict['chars_count_lm'] += chars_count_lm_
                
        del all_probs, preds, preds_tensor, pred_text, ch_audios, ch_targets
                
    print('\n', flush=True)
    logging.info('Greedy WER/CER = {:.2%}/{:.2%}'.format(wer_dist_greedy / words_count_greedy, cer_dist_greedy / chars_count_greedy))
    
    if args.decoding_mode in ["beamsearch_ngram", "beamsearch"]:
        for hp in hp_grid:
            res_dict = hp_grid_results[tuple(hp.values())]
            logging.info(f"Beam search decoding with width={hp['beam_width']}, alpha={hp['beam_alpha']}, beta={hp['beam_beta']}")
            
            if lm_path:
                logging.info(
                    'WER/CER with beam search decoding and N-gram model = {:.2%}/{:.2%}'.format(
                        res_dict['wer_dist_first'] / res_dict['words_count_lm'], res_dict['cer_dist_first'] / res_dict['chars_count_lm']
                    )
                )
            else:
                logging.info(
                    'WER/CER with beam search decoding = {:.2%}/{:.2%}'.format(
                        res_dict['wer_dist_first'] / res_dict['words_count_lm'], res_dict['cer_dist_first'] / res_dict['chars_count_lm']
                    )
                )
            logging.info(
                'Oracle WER/CER in candidates with perfect LM= {:.2%}/{:.2%}'.format(
                    res_dict['wer_dist_best'] / res_dict['words_count_lm'], res_dict['cer_dist_best'] / res_dict['chars_count_lm']
                )
            )
            logging.info("=================================================================================")
            
            res_dict['wer_score'] = res_dict['wer_dist_first'] / res_dict['words_count_lm']
            res_dict['cer_score'] = res_dict['cer_dist_first'] / res_dict['chars_count_lm']
            
            if res_dict['out_file']:
                logging.info(f"Stored the predictions of beam search decoding at: {res_dict['out_file'].name}")
                res_dict['out_file'].close()
    
    if args.decoding_mode in ["beamsearch_ngram", "beamsearch"] and len(hp_grid) > 1:
        cer_scores = [x['cer_score'] for x in hp_grid_results.values()]
        best_key = hp_grid[np.argmin(cer_scores)]
        best_value = hp_grid_results[tuple(best_key.values())]
        print('\n', flush=True)
        logging.info(f"Grid search winner: width={best_key['beam_width']}, alpha={best_key['beam_alpha']}, beta={best_key['beam_beta']}")
        logging.info("With WER/CER = {:.2%}/{:.2%}".format(best_value['wer_score'], best_value['cer_score']))


if __name__ == '__main__':
    main()
