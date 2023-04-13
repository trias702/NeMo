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

import argparse
import logging
import os
import multiprocessing
import json
import numpy as np
from functools import partial
from tqdm import tqdm
import soundfile as sf
import librosa


def read_manifest(manifest):    
    with open(manifest, 'r', encoding='utf_8') as fr:
        data = [json.loads(line) for line in fr]
    
    return data


def parse_args():
    parser = argparse.ArgumentParser(description='Create synthetic code-switched data manifest and audio from monolingual data manifests')
    
    parser.register('type', 'bool', (lambda x: x.lower() in ("true", "1")))
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    
    required.add_argument('-m', '--manifests',
                        required=True,
                        nargs='+',
                        type=str,
                        help='path to monolingual manifests')
    required.add_argument('-o', '--output',
                        required=True,
                        type=str,
                        help='path to directory for saving all content')
    required.add_argument('-l', '--languages',
                        required=True,
                        nargs='+',
                        type=str,
                        help='list of languages corresponding to each manifest')
    
    optional.add_argument('--min_duration',
                        default=16,
                        type=int,
                        help='minimum duration (secs) of each synthetic utterance')
    optional.add_argument('--max_duration',
                        default=20,
                        type=int,
                        help='maximum duration (secs) of each synthetic utterance')
    optional.add_argument('--min_mono',
                        default=0.2,
                        type=float,
                        help='percentage of output data which is guaranteed to be monolingual')
    optional.add_argument('--req_hours',
                        default=100,
                        type=int,
                        help='the number of hours required for the synthetic manifest')
    optional.add_argument('--lang_probs',
                        nargs='+',
                        type=float,
                        help='probabilities to sample from for each language')
    optional.add_argument('--seed',
                        default=5831445,
                        type=int,
                        help='random seed to use')
    optional.add_argument('--db_norm',
                        default=-25.0,
                        type=float,
                        help='DB level to normalise all audio to')
    optional.add_argument('--pause_start',
                        default=20,
                        type=int,
                        help='pause to be added at the beginning of the sample (msec)')
    optional.add_argument('--pause_join',
                        default=100,
                        type=int,
                        help='pause to be added between different phrases of the sample (msec)')
    optional.add_argument('--pause_end',
                        default=20,
                        type=int,
                        help='pause to be added at the end of the sample (msec)')
    optional.add_argument('--sample_rate',
                        default=16000,
                        type=int,
                        help='sample rate for generated audio')
    optional.add_argument('--pure_random',
                        action='store_true',
                        help='whether to draw pure randomly on each pass')
    optional.add_argument('--num_workers',
                        type=int,
                        default=int(os.cpu_count() * 0.75),
                        help="No of multiprocessing workers, Defaults to os.cpu_count() * 0.75")
    
    args = parser.parse_args()
    
    return args


def create_cs_manifest(args, manifests):
    total_duration = 0
    constructed_data = []
    sample_id = 1
    
    langs = args.languages
    langs_set = set(langs)
    N = len(langs)
    num_samples = {lang:len(manifests[lang]['data']) for lang in manifests.keys()}
    if args.lang_probs:
        prob_dict = {l:args.lang_probs[i] for i,l in enumerate(langs)}
    else:
        prob_dict = {l:1.0/N for l in langs}

    while total_duration < (args.req_hours * 3600.0):
        created_sample_duration_sec = 0
        created_sample_dict = {}
        created_sample_dict['lang_ids'] = []
        created_sample_dict['texts'] = []
        created_sample_dict['paths'] = []
        created_sample_dict['durations'] = []
        
        pure_mono = np.random.rand() <= args.min_mono

        while created_sample_duration_sec < args.min_duration:
            if (args.pure_random and not pure_mono) or (len(set(created_sample_dict['lang_ids'])) == 0 or len(set(created_sample_dict['lang_ids'])) == N):
                lang_id = np.random.choice(langs, p=args.lang_probs)
            #elif pure_mono:
            #    lang_id = created_sample_dict['lang_ids'][0]
            else:
                p = np.array(list(map(prob_dict.get, list(langs_set - set(created_sample_dict['lang_ids'])))))
                p = p / p.sum()
                lang_id = np.random.choice(list(langs_set - set(created_sample_dict['lang_ids'])), p=p)
            sample = manifests[lang_id]['data'][np.random.randint(0, num_samples[lang_id])]
            sample_path = os.path.dirname(manifests[lang_id]['path'])

            if (created_sample_duration_sec + sample['duration']) > args.max_duration:
                continue
            if not os.path.exists(os.path.join(sample_path, sample['audio_filepath'])):
                logging.error(f"Cannot locate path on disk: {os.path.join(sample_path, sample['audio_filepath'])}")
                continue
            
            created_sample_duration_sec += sample['duration']
            created_sample_dict['lang_ids'].append(lang_id)
            created_sample_dict['texts'].append(sample['text'])
            created_sample_dict['paths'].append(os.path.join(sample_path, sample['audio_filepath']))
            created_sample_dict['durations'].append(sample['duration'])
            
            if pure_mono:
                break
        
        if pure_mono:
            assert len(set(created_sample_dict['lang_ids'])) == 1, "pure_mono failed"

        created_sample_dict['total_duration'] = created_sample_duration_sec

        created_sample_dict['uid'] = sample_id
        sample_id += 1

        constructed_data.append(created_sample_dict)
        total_duration += created_sample_duration_sec

    return constructed_data


def mp_create_single(row, args=None):
    comp = np.zeros(int(args.pause_start * args.sample_rate / 1000.0))
    
    for idx, path in enumerate(row['paths']):
        if not os.path.exists(path):
            return None
        
        wav, sr = sf.read(path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.T
            wav = wav.sum(axis=0)/wav.shape[0]
        wav = np.trim_zeros(wav)
        if sr != args.sample_rate:
            wav = librosa.core.resample(wav, orig_sr=sr, target_sr=args.sample_rate)
        
        wav_norm = wav * (10.0 ** (args.db_norm / 20.0) / np.maximum(0.01, (wav ** 2).mean() ** 0.5))
        
        if idx < len(row['paths']) - 1:
            wav_norm = np.append(wav_norm, np.zeros(int(args.pause_join * args.sample_rate / 1000.0)))
        
        comp = np.append(comp, wav_norm)
    
    comp = np.append(comp, np.zeros(int(args.pause_end * args.sample_rate / 1000.0)))
    duration = len(comp) / args.sample_rate
    comp_path = os.path.join(args.output, 'Audio', str(row['uid']) + '_' + ('_'.join(row['lang_ids'])) + '_' + ('_'.join([str(x) for x in row['durations']])) + '.wav')
    
    sf.write(comp_path, comp, args.sample_rate, format='WAV')
    
    entry = {}
    entry['audio_filepath'] = comp_path
    entry['duration'] = duration
    entry['text'] = [{'lang': lang, 'str': text} for lang, text in zip(row['lang_ids'], row['texts'])]
    
    return entry


def main():
    args = parse_args()

    if len(args.manifests) != len(args.languages):
        logging.error('Number of manifests passed does not match number of languages passed. Aborting.')
        exit(255)
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'Audio'), exist_ok=True)
    
    np.random.seed(args.seed)

    logging.info('Reading manifests')
    manifests = {lang: {'path': file, 'data': read_manifest(file)} for lang,file in zip(args.languages, args.manifests)}

    logging.info('Creating CS manifest')
    constructed_data = create_cs_manifest(args, manifests)

    manifest_save_path = os.path.join(args.output, 'synth_cs_manifest.json')
    
    process_partial = partial(mp_create_single, args=args)
    with open(manifest_save_path, 'w', encoding='utf_8', newline='\n') as manifest:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for result in tqdm(pool.imap(func=process_partial, iterable=constructed_data), total=len(constructed_data)):
                if result is not None:
                    manifest.write(json.dumps(result, ensure_ascii=False) + '\n')

    print("Synthetic CS manifest saved at :", manifest_save_path, flush=True)
    print('All done!', flush=True)


if __name__ == "__main__":
    main()
