import codecs
import json
import os


#Copyright (C) 2020 Panagiotis Louridas <louridas@aueb.gr>

CONSONANTS = set(['β', 'γ', 'δ', 'ζ', 'κ', 'λ', 'μ', 'ν',
                  'ξ', 'π', 'ρ', 'σ', 'τ', 'φ', ' χ', 'ψ'])

ACCENTED_VOWELS = set(['ά', 'έ', 'ή', 'ί', 'ό', 'ύ', 'ώ'])


def make_genitive(noun):
    # male cases
    if len(noun) == 0:
        return ''
    if noun[-1:] == 'μ' or noun[-1:] == 'τ':
        return noun
    if noun[-3:] == 'έων':
        return noun[:-3] + 'έοντα'
    if noun[-2:] == 'ών':
        if noun[-3] in CONSONANTS:
            return noun[:-2] + 'ώντα'
        else:
            return noun

    # ων, ωνα
    if noun[-2:] == 'ός':
        return noun[:-2] + 'ού'
    if noun[-2:] == 'ος':  # only treat male, not female like διάμετρος
        return noun[:-2] + 'ου'  # not always, e.g., άγγελος / αγγέλου
    if noun[-1] == 'ς':
        return noun[:-1]
    if noun[-1] == 'λ':
        return noun

    # female cases
    return noun + 'ς'


def make_accusative(noun):
    if len(noun) == 0:
        return ''
    if noun[-3:] == 'έων':
        return noun
    if noun[-1] != 'ς':
        return noun
    return noun[:-1]


def make_vocative(noun):
    if len(noun) == 0:
        return ''
    if noun[-3:] == 'έων':
        return noun
    if noun[-1] != 'ς':
        return noun
    if noun.endswith('ος'):
        i = 3
        # if no accented vowel before suffix we just drop final s
        if noun[-i] not in CONSONANTS:
            if noun[-i] not in ACCENTED_VOWELS:
                return noun[:-1]

        # otherwise skip consonants to find the previous syllable
        noun_len = len(noun)
        while (i < noun_len and noun[-i] in CONSONANTS):
            i += 1

        # if accented we just drop final s
        if noun[-i] in ACCENTED_VOWELS:
            return noun[:-1]

        # if not accented and ends with 'ος' we change to 'ε'
        else:
            return noun[:-2] + 'ε'
    else:
        return noun[:-1]


def find_case(name, case, file):
    if file == 'female_surname_cases_fast.json':
        return name
    else:
        if case == 'ονομαστική':
            return name
        elif case == 'γενική':
            return make_genitive(name)
        elif case == 'αιτιατική':
            return make_accusative(name)
        elif case == 'κλητική':
            return make_vocative(name)


path_to_files = '../out_files/wiki_data/'
file_names = ['male_name_cases.json',
              'female_name_cases.json',
              'male_surname_cases.json',
              'female_surname_cases.json',
              ]

for file in file_names:

    # Produce cases from nominative for those without cases
    with codecs.open(os.path.join(path_to_files, file)) as f:
        parsed = f.read()
        names_dict = json.loads(parsed)

    for name, info in names_dict.items():

        cases = info['ενικός']

        for case, value in cases.items():

            if value == '':
                produced_value = find_case(name, case, file)
                # replace value
                names_dict[name]['ενικός'][case] = produced_value

    out_file = os.path.join(path_to_files,
                            os.path.splitext(file)[0] + '_populated.json')

    with codecs.open(out_file, 'w', encoding='utf-8') as f:
        json.dump(names_dict, f, ensure_ascii=False, indent=4)
