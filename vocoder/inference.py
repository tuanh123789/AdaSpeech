from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import glob
import json
import torch
import argparse
import numpy as np

from scipy.io.wavfile import write

from src.preprocessing.meldataset import MAX_WAV_VALUE
from models.hifigan import Generator
from modules.env import AttrDict
from modules.denoiser import Denoiser

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    if a.denoiser_strength > 0:
        if torch.cuda.is_available():
            denoiser = Denoiser(generator).cuda()
        else:
            denoiser = Denoiser(generator).cpu()

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_mels_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            x = np.load(os.path.join(a.input_mels_dir, filename))
            x = torch.FloatTensor(x).to(device)
            y_g_hat = generator(x)

            if a.denoiser_strength > 0:
                audio = denoiser(y_g_hat.squeeze(0), a.denoiser_strength)
            else:
                audio = y_g_hat.unsqueeze(0)
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print('Audio {} Saved: {}'.format(output_file,
                                              time.strftime('%H:%M:%S', time.gmtime(audio.shape[0] / h.sampling_rate))))


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', required=True, help='Path to mel-spectrogram folder')
    parser.add_argument('--output_dir', default='data/generated')
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

