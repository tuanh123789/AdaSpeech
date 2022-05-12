import json
from utils import g2p_convert
from tqdm import tqdm

corpus = []
with open('all_syllable.txt', encoding='utf-8') as file:
    for line in file.readlines():
        corpus.append(line.replace('\n',''))

if __name__ == "__main__":
    ipa_corpus = {}
    for word in tqdm(corpus):
        phone = g2p_convert(word)
        phone[-1] = phone[-1]
        ipa_corpus[word] = ' '.join(phone)
    with open('dict_phoneme.json', 'w', encoding='utf-8') as file:
        json.dump(ipa_corpus, file, ensure_ascii=False, indent=4)
