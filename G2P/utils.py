from pandas.core.common import flatten
import string

_pad = ['<pad>']
_silent = ['<silent>']
_eos = ['<s>', '</s>']

_consonants = ['HH', 'M', 'S_vn', 'K', 'CH_vn', 'Z', 'T_vn', 'NG_vn', 'L', 'KH_vn', 'N', 'Y',
               'TR_vn', 'S', 'D', 'TH_vn', 'P', 'F', 'NH_vn', 'KW_vn', 'B', 'G_vn', 'V']  # 23

_tone = ['0', '1', '2', '3', '4', '5']
_medial = ['WU_vn', 'WO_vn']  # 2
_monophthongs = ['AW_vn', 'EE_vn', 'EH1', 'AA_vn', 'OW_vn', 'IY1', 'UW_vn', 'OO_vn', 'UW1', 'O_vn', 'AO1', 'AE1']  # 12

_diphthong = ['IE_vn', 'UO_vn', 'WA_vn']  # 3

_coda = ['NH_vn', 'M', 'N', 'IZ_vn', 'K', 'UZ_vn', 'YZ_vn', 'P', 'OZ_vn', 'T', 'NG']  # 12
_letters = _consonants + _medial + _monophthongs + _diphthong + _coda + _tone

# Export all symbols:
symbols = _pad + _letters + _silent + _eos
symbols = list(flatten(
    [['B-{}'.format(s), 'I-{}'.format(s), 'E-{}'.format(s)] if s not in _pad + _tone + _eos
     else [s] for s in symbols]))

symbols = [_.upper() for _ in symbols]

g2p_consonants = {'b': 'B', 'ch': 'CH_vn', 'đ': 'D', 'ph': 'F', 'h': 'HH', 'd': 'Y', 'k': 'K', 'qu': 'KW_vn', 'q': 'K',
                  'c': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'nh': 'NH_vn', 'ng': 'NG_vn', 'ngh': 'NG_vn', 'p': 'P', 'x': 'S',
                  's': 'S_vn', 't': 'T_vn', 'th': 'TH_vn', 'tr': 'TR_vn', 'v': 'V', 'kh': 'KH_vn', 'g': 'G_vn', 'gh': 'G_vn', 'gi': 'Y',
                  'r': 'Z'}
g2p_medial = {'u': 'WU_vn', 'o': 'WO_vn'}
g2p_monophthongs = {'ă': 'AW_vn', 'ê': 'EE_vn', 'e': 'EH1', 'â': 'AA_vn', 'ơ': 'OW_vn', 'y': 'IY1', 'i': 'IY1', 'ư': 'UW_vn', 'ô': 'OO_vn',
                    'u': 'UW1', 'oo': 'O_vn', 'o': 'AO1', 'a': 'AE1'}
g2p_diphthongs = {'yê': 'IE_vn', 'iê': 'IE_vn', 'ya': 'IE_vn', 'ia': 'IE_vn', 'ươ': 'WA_vn', 'ưa': 'WA_vn', 'uô': 'UO_vn', 'ua': 'UO_vn'}
g2p_coda = {'m': 'M', 'n': 'N', 'ng': 'NG', 'nh': 'NH_vn', 'p': 'P', 't': 'T', 'ch': 'K', 'k': 'K', 'c': 'K',
            'u': 'UZ_vn', 'o': 'OZ_vn', 'y': 'YZ_vn', 'i': 'IZ_vn'}

g2p_tone = {u'á': 1, u'à': 2, u'ả': 3, u'ã': 4, u'ạ': 5,
            u'ấ': 1, u'ầ': 2, u'ẩ': 3, u'ẫ': 4, u'ậ': 5,
            u'ắ': 1, u'ằ': 2, u'ẳ': 3, u'ẵ': 4, u'ặ': 5,
            u'é': 1, u'è': 2, u'ẻ': 3, u'ẽ': 4, u'ẹ': 5,
            u'ế': 1, u'ề': 2, u'ể': 3, u'ễ': 4, u'ệ': 5,
            u'í': 1, u'ì': 2, u'ỉ': 3, u'ĩ': 4, u'ị': 5,
            u'ó': 1, u'ò': 2, u'ỏ': 3, u'õ': 4, u'ọ': 5,
            u'ố': 1, u'ồ': 2, u'ổ': 3, u'ỗ': 4, u'ộ': 5,
            u'ớ': 1, u'ờ': 2, u'ở': 3, u'ỡ': 4, u'ợ': 5,
            u'ú': 1, u'ù': 2, u'ủ': 3, u'ũ': 4, u'ụ': 5,
            u'ứ': 1, u'ừ': 2, u'ử': 3, u'ữ': 4, u'ự': 5,
            u'ý': 1, u'ỳ': 2, u'ỷ': 3, u'ỹ': 4, u'ỵ': 5,
            }
remove_tone = {u'á': u'a', u'à': u'a', u'ả': u'a', u'ã': u'a', u'ạ': u'a',
               u'ấ': u'â', u'ầ': u'â', u'ẩ': u'â', u'ẫ': u'â', u'ậ': u'â',
               u'ắ': u'ă', u'ằ': u'ă', u'ẳ': u'ă', u'ẵ': u'ă', u'ặ': u'ă',
               u'é': u'e', u'è': u'e', u'ẻ': u'e', u'ẽ': u'e', u'ẹ': u'e',
               u'ế': u'ê', u'ề': u'ê', u'ể': u'ê', u'ễ': u'ê', u'ệ': u'ê',
               u'í': u'i', u'ì': u'i', u'ỉ': u'i', u'ĩ': u'i', u'ị': u'i',
               u'ó': u'o', u'ò': u'o', u'ỏ': u'o', u'õ': u'o', u'ọ': u'o',
               u'ố': u'ô', u'ồ': u'ô', u'ổ': u'ô', u'ỗ': u'ô', u'ộ': u'ô',
               u'ớ': u'ơ', u'ờ': u'ơ', u'ở': u'ơ', u'ỡ': u'ơ', u'ợ': u'ơ',
               u'ú': u'u', u'ù': u'u', u'ủ': u'u', u'ũ': u'u', u'ụ': u'u',
               u'ứ': u'ư', u'ừ': u'ư', u'ử': u'ư', u'ữ': u'ư', u'ự': u'ư',
               u'ý': u'y', u'ỳ': u'y', u'ỷ': u'y', u'ỹ': u'y', u'ỵ': u'y',
               }


# code
def g2p_convert(g_word):
    """Tone location: Location of tone in phonemes of word
    input form: {inside, last, both}
    Two type of phonemes:
    - Tone at end of syllable: C1wVC2T
    - Tone after vowel: C1wVTC2
    - Tone present both: C1wVTC2T
    """

    p_word = []
    raw_g_word = g_word
    g_word = list(g_word)

    # tone detection
    tone = '0'
    for i, w in enumerate(g_word):
        if w in g2p_tone:
            tone = str(g2p_tone[w])
            g_word[i] = remove_tone[w]
            break
    g_word = ''.join(g_word)
    if g_word.startswith('giê'):
        g_word = 'd' + g_word[1:]
        raw_g_word = 'd' + raw_g_word[1:]

    # convert graphemes to phonemes
    if len(g_word) == 1:
        # từ 1 âm tiết
        p_word.append(g2p_monophthongs[g_word])
    elif len(g_word) == 2:
        # Từ 2 âm tiết
        # Trường hợp đặc biệt: gi
        if g_word == 'gi':
            p_word = ['Y']
        # Tồn tại C1
        elif g_word[0] in g2p_consonants:
            p_word.extend([g2p_consonants[g_word[0]], g2p_monophthongs[g_word[1]]])
        # Tồn tại C2
        elif g_word[1] in g2p_coda:
            if g_word[0] == 'o':
                if g_word[1] in ['n', 't', 'i']:
                    p_word.extend(['O_vn', g2p_coda[g_word[1]]])
                else:
                    p_word.extend(['AO1', g2p_coda[g_word[1]]])
            else:
                p_word.extend([g2p_monophthongs[g_word[0]], g2p_coda[g_word[1]]])
        # Không tồn tại C
        else:
            # là dịphthong?
            if g_word in g2p_diphthongs:
                p_word.append(g2p_diphthongs[g_word])
            else:
                p_word.extend([g2p_medial[g_word[0]], g2p_monophthongs[g_word[1]]])
    elif len(g_word) == 3:
        # Từ 3 âm tiết
        # Trường hợp đặc biệt nguyên âm oo
        if 'oo' in g_word:
            if g_word[:2] == 'oo':
                p_word.extend(['O_vn', g2p_coda[g_word[2]]])
            else:
                p_word.extend([g2p_consonants[0], 'O_vn'])
        else:
            # C1 có 2 âm tiết
            if g_word[:2] in g2p_consonants:
                if g_word[:2] in ['gi', 'qu'] and g_word[2] in g2p_coda:
                    if g_word[2] in ['i', 'u']:
                        p_word.extend([g2p_consonants[g_word[:2]], g2p_monophthongs[g_word[2]]])
                    else:
                        p_word.extend([g2p_consonants[g_word[0]], g2p_monophthongs[g_word[1]], g2p_coda[g_word[2]]])
                else:
                    p_word.extend([g2p_consonants[g_word[:2]], g2p_monophthongs[g_word[2]]])
            # C1 có 1 âm tiết
            elif g_word[0] in g2p_consonants:
                # C1 + diphthong
                if g_word[1:] in g2p_diphthongs:
                    p_word.extend([g2p_consonants[g_word[0]], g2p_diphthongs[g_word[1:]]])
                # C1 + monophthongs + C2
                elif g_word[2] in g2p_coda:
                    if g_word[1] == 'o':
                        if g_word[1] in ['n', 't', 'i']:
                            p_word.extend([g2p_consonants[g_word[0]], 'O_vn', g2p_coda[g_word[2]]])
                        else:
                            p_word.extend([g2p_consonants[g_word[0]], 'AO1', g2p_coda[g_word[2]]])
                    else:
                        p_word.extend([g2p_consonants[g_word[0]], g2p_monophthongs[g_word[1]], g2p_coda[g_word[2]]])
                else:
                    p_word.extend([g2p_consonants[g_word[0]], g2p_medial[g_word[1]], g2p_monophthongs[g_word[2]]])
            else:
                # C2 có 2 âm tiết
                if g_word[1:] in g2p_coda:
                    if g_word[0] == 'o':
                        p_word.extend(['AO1', g2p_coda[g_word[1:]]])
                    else:
                        p_word.extend([g2p_monophthongs[g_word[0]], g2p_coda[g_word[1:]]])
                # C2 có 1 âm tiết
                elif g_word[2] in g2p_coda:
                    if g_word[:2] in g2p_diphthongs:
                        p_word.extend([g2p_diphthongs[g_word[:2]], g2p_coda[g_word[2]]])
                    else:
                        if g_word[1] == 'o':
                            if g_word[1] in ['n', 't', 'i']:
                                p_word.extend([g2p_medial[g_word[0]], 'O_vn', g2p_coda[g_word[2]]])
                            else:
                                p_word.extend([g2p_medial[g_word[0]], 'AO1', g2p_coda[g_word[2]]])

                        else:
                            p_word.extend([g2p_medial[g_word[0]], g2p_monophthongs[g_word[1]], g2p_coda[g_word[2]]])
                # w + V
                else:
                    p_word.extend([g2p_medial[g_word[0]], g2p_diphthongs[g_word[1:]]])
    else:
        g_word = list(g_word)
        # Tìm C1
        if raw_g_word[:3] in g2p_consonants:
            C1 = raw_g_word[:3]
            del g_word[:3]
        elif raw_g_word[:2] in g2p_consonants:
            C1 = raw_g_word[:2]
            del g_word[:2]
        elif raw_g_word[0] in g2p_consonants:
            C1 = raw_g_word[0]
            del g_word[0]
        else:
            C1 = ''

        # Tìm C2
        if raw_g_word[-3:] in g2p_coda:
            C2 = raw_g_word[-3:]
            del g_word[-3:]
        elif raw_g_word[-2:] in g2p_coda:
            C2 = raw_g_word[-2:]
            del g_word[-2:]
        elif raw_g_word[-1] in g2p_coda:
            C2 = raw_g_word[-1]
            del g_word[-1]
        else:
            C2 = ''

        # Tách V
        g_word = ''.join(g_word)
        V = []
        if len(g_word) == 3:
            V.extend([g2p_medial[g_word[0]], g2p_diphthongs[g_word[-2:]]])
        elif len(g_word) == 2:
            if g_word == 'oo':
                V.append('O_vn')
            elif g_word in g2p_diphthongs:
                V.append(g2p_diphthongs[g_word])
            else:
                V.extend([g2p_medial[g_word[0]], g2p_monophthongs[g_word[1]]])
        else:
            if g_word:
                if g_word == 'o':
                    if C2 in ['n', 't', 'i']:
                        V.append('O_vn')
                    else:
                        V.append('AO1')
                else:
                    V.append(g2p_monophthongs[g_word])
            else:
                if C1 == 'gi':
                    C1 = 'g'
                    V = ['IY1']
                elif C1 == 'qu':
                    C1 = 'q'
                    V = ['u']
                elif C2 == 'i' or C2 == 'y':
                    V = ['IY1']
                    C2 = ''
                else:
                    print('error')
                    print(raw_g_word)
                    exit()
        if C1:
            C1 = g2p_consonants[C1]
        if C2:
            C2 = g2p_coda[C2]
        p_word.extend([_ for _ in list(flatten([C1, V, C2])) if _])

    p_word.append(tone)

    return p_word