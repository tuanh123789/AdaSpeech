#from vncorenlp import VnCoreNLP
from pyvi import ViTokenizer
from g2p_en import G2p
import json


g2p = G2p()
dict_phoneme = json.load(open('G2P/dict_phoneme.json','r'))
all_syllable = list(dict_phoneme.keys())

def convert_text_to_ipa(text):
    segment_text = ViTokenizer.tokenize(text)
    segment_text = segment_text.split()
    print(segment_text)
    convert_text = []
    for sentence in [segment_text]:
        convert_sentence = []
        for word in sentence:
            if '_' in word:
                for sub_word in word.replace('_',' ').split():
                    convert_sentence.append(sub_word)
            else:
                convert_sentence.append(word)
        sub_cv = []
        eng_pos = []
        for sym in convert_sentence:
            if sym in all_syllable:
                for ph in dict_phoneme[sym].split():
                    sub_cv.append(ph)
            else:
                phone_eng = g2p(sym)
                eng_pos.append((len(sub_cv), len(sub_cv)+len(phone_eng)))
                for ph in phone_eng:
                    sub_cv.append(ph)

    convert_text.append(" ".join(sub_cv))
    convert_text = ' '.join(convert_text)
    return convert_text, eng_pos

