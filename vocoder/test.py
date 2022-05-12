from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import json
import glob
import torch
import glob
import argparse


from scipy.io.wavfile import write

from modules.env import AttrDict
from src.preprocessing.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models.hifigan import Generator
from modules.denoiser import Denoiser


h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    if a.denoiser_strength > 0:
        if torch.cuda.is_available():
            denoiser = Denoiser(generator).cuda()
        else:
            denoiser = Denoiser(generator).cpu()

    # f = open(a.input_wavs_dir, 'r', encoding='utf8')
    # filelist = f.read().split('\n')
    filelist = glob.glob(a.test_file + '/*.wav')
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(filename)
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))

            y_g_hat = generator(x)

            if a.denoiser_strength > 0:
                audio = denoiser(y_g_hat.squeeze(0), a.denoiser_strength)
            else:
                audio = y_g_hat.unsqueeze(0)
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            # output_file = os.path.join(a.output_dir, filename.split('/')[-1].replace('.wav', '_generated.wav'))
            output_file = os.path.join(a.output_dir, filename.split('/')[-1])
            write(output_file, h.sampling_rate, audio)
            print('Audio {} Saved: {}'.format(output_file,
                                              time.strftime('%H:%M:%S', time.gmtime(audio.shape[-1] / h.sampling_rate))))


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--output_dir', default='data/generated/')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--denoiser_strength', default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    h.fmax = 11025

    torch.manual_seed(h.seed)

    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
