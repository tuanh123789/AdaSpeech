_pad = "_"
_punctuation = ",."
_consonants = ['HH', 'M', 'S_vn', 'K', 'CH_vn', 'Z', 'T_vn', 'NG_vn', 'L', 'KH_vn', 'N', 'Y',
        'TR_vn', 'S', 'D', 'TH_vn', 'P', 'F', 'NH_vn', 'KW_vn', 'B', 'G_vn', 'V', 'd']

_tone = ['0', '1', '2', '3', '4', '5']
_medial = ['WU_vn', 'WO_vn'] 
_monophthongs = ['AW_vn', 'EE_vn', 'EH1', 'AA_vn', 'OW_vn', 'IY1', 'UW_vn', 'OO_vn', 'UW1', 'O_vn', 'AO1', 'AE1']

_diphthong = ['IE_vn', 'UO_vn', 'WA_vn']

_coda = ['NH_vn', 'M', 'N', 'IZ_vn', 'K', 'UZ_vn', 'YZ_vn', 'P', 'OZ_vn', 'T', 'NG'] 

_eng_phoneme = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                'EY2', 'F', 'G', 'HH',
                'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                'M', 'N', 'NG', 'OW0', 'OW1',
                'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                'UH0', 'UH1', 'UH2', 'UW',
                'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
                ]

_silences = ["sp", "spn", "sil"]

all_symbols = (
    [_pad]
    + list(_punctuation)
    + _consonants
    + _tone
    + _medial
    + _monophthongs
    + _diphthong
    + _coda
    + _silences
    + _eng_phoneme
)

symbols = []
for sym in all_symbols:
    if sym not in symbols:
        symbols.append(sym)