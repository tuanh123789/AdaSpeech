import os
import json
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm

from meldataset import MAX_WAV_VALUE, mel_spectrogram
from audio_processing import load_wav


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dirs', required=True, help='path to audio dataset')
    parser.add_argument('--split_percent', default=0.9, help='percent to split train/test')
    parser.add_argument('--config', default='config/config_v1.json', help='config to generate mel-spectrogram')
    parser.add_argument('--use_load_from_disk', default=False)

    args = parser.parse_args()

    wav_file = [file_name for file_name in glob.glob(os.path.join(args.wav_dirs, 'wavs') + '/*.wav')]
    random.shuffle(wav_file)

    if args.use_load_from_disk:
        f = open(args.config, 'r')
        config = json.load(f)
        mel_path = os.path.join(args.wav_dirs, 'mels')
        os.makedirs(mel_path, exist_ok=True)
        for filename in tqdm(wav_file, desc='generating mel-spectrogram'):
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if sampling_rate != config['sampling_rate']:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, config['sampling_rate']))
            mel = mel_spectrogram(audio, n_fft=config['n_fft'], num_mels=config['num_mels'], sampling_rate=config['sampling_rate']
                                  , hop_size=config['hop_size'], win_size=config['win_size'], fmin=config['fmin'], fmax=config['fmax'])
            np.save(os.path.join(mel_path, filename.split('/')[-1] + '.npy'), mel)

    print("Total {} wav file".format(len(wav_file)))
    with open('data/train_files.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(wav_file[:int(args.split_percent * len(wav_file))]))
    with open('data/test_files.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(wav_file[int(args.split_percent * len(wav_file)):]))
