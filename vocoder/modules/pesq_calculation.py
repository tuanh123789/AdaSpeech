import os
import json
import argparse


from pesq import pesq
from scipy.io import wavfile
from multiprocessing import Pool


def calculate(inputs):
    ground_truth, predicted = inputs

    rate, ref = wavfile.read(ground_truth)
    rate, deg = wavfile.read(predicted)

    return [ground_truth.split('/')[-1], pesq(16000, ref, deg, 'wb')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden_waves', required=True)
    parser.add_argument('--predicted_waves', required=True)
    a = parser.parse_args()

    waves = []
    for wav in os.listdir(a.golden_waves):
        if os.path.isfile(wav.replace(a.golden_waves, a.predicted_waves)):
            waves.append([wav, wav.replace(a.golden_waves, a.predicted_waves)])

    p = Pool(48)
    output = p.map(calculate, waves)

    f = open(os.path.join(a.predicted_waves, 'pesq_score.json'), 'w', encoding='utf8')
    json.dump(output, f, ensure_ascii=False, indent=4)
