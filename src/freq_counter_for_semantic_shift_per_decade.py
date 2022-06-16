# -*- coding: utf-8 -*-
import unicodedata
from collections import defaultdict
import re
from collections import Counter
import pandas as pd
import datetime

print(datetime.datetime.now().time())

# \u00b7 is middle dot
# \u0387 is Greek ano teleia
punct_regex = re.compile(r"([?.!,;\u00b7\u0387])")

def str_clean(s):
   normalized = ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')
   separate_punct = re.sub(punct_regex, r" \1 ", normalized)
   collapse_spaces  = re.sub(r'\s+', " ", separate_punct)
   lc = collapse_spaces.lower()
   return lc

word_freq = defaultdict(int)

df = pd.read_csv('../out_files/tell_all_cleaned.csv') #, nrows=10
print('read input file')
df = df[df['speech'].notna()]
print('cleaning speeches...')
df.speech = df.speech.apply(lambda x: str_clean(x))
print('speeches cleaned')

df.speech = df.speech.apply(lambda x: x.replace(".", " "))
df.sitting_date = pd.to_datetime(df.sitting_date, format="%d/%m/%Y")

#New column year
df['year'] = df['sitting_date'].dt.year
df['decade'] = (df['year']//10)*10
# df = df[df.decade != 1980] # remove dates before 2000 to catch the three last decades
df = df[df.decade != 2020]# remove dates after 2019 to catch the three last decades
df = df[df.decade != 2000]# remove dates after 2019 to catch the three last decades

PERdecade_df = df.groupby(df.decade)['speech'].apply(' '.join).reset_index()

for decade in PERdecade_df.decade.to_list():
    print(decade)
    subdf = PERdecade_df.loc[(PERdecade_df.decade==decade)]

    tell_all = subdf.speech.iloc[0].lower()

    # tell_all = re.sub("\d+", "", tell_all)
    tell_all = re.sub("\s\s+" , ' ', tell_all)

    freqs = Counter()
    subdf.speech.apply(lambda x: freqs.update(x.split()))
    print('finished counting')
    total_number = sum(freqs.values())
    print('total number of tokens:', total_number)

    freqs_df = pd.DataFrame.from_dict(freqs, orient='index',
                                      columns=['frequency'])
    freqs_df = freqs_df.reset_index()

    freqs_df = freqs_df.rename(columns={'index': 'word'})
    mask = (freqs_df['word'].str.len() > 1)
    freqs_df = freqs_df.loc[mask]
    print('Removed entries with one character.')

    freqs_df = freqs_df.sort_values('frequency').reset_index(drop=True)

    freqs_df['percentage'] = freqs_df['frequency'] / total_number

    freqs_df.to_csv('../out_files/freqs_for_semantic_shift_cleaned_data_decade'+str(decade)+'.csv', index=False)

print(datetime.datetime.now().time())