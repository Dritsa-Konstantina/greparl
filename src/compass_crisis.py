#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import random as rn
import datetime
import re
import os
from cade.cade import CADE
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from tqdm import tqdm
import tqdm.notebook as tq
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tabulate import tabulate

def compute_cosine_similarity(model1,model2,word):
    vector1 = model1.wv[word].reshape(1,-1)
    vector2 = model2.wv[word].reshape(1,-1)
    return(cosine_similarity(X=vector1, Y=vector2)[0][0])

def step_one_pairs(list_of_items):
    return [(list_of_items[i],list_of_items[i+1]) for i in range(len(list_of_items)-1)]


tqdm.pandas()

parser = ArgumentParser()
parser.add_argument("-d", "--diter", type=int)
parser.add_argument("-s", "--siter", type=int)
parser.add_argument("-z", "--size", type=int)
parser.add_argument("-n", "--rows", type=str)

args = vars(parser.parse_args())
diter = args['diter']
siter = args['siter']
vector_size = args['size']
no_rows = args['rows']

if no_rows.isdecimal():
    no_rows = int(no_rows)

df = pd.read_csv('../out_files/tell_all_cleaned.csv')

df = df[df['speech'].notna()]

# New column year
df.sitting_date = pd.to_datetime(df.sitting_date, format="%d/%m/%Y") 
df['year'] = df['sitting_date'].dt.year

# Group speeches by year
'''
cade tool uses gensim.models.word2vec.LineSentence() to iterate over the training corpus
gensim.models.word2vec.LineSentence() takes as input 
a file that contains sentences: one line = one sentence.
Words must be already preprocessed and separated by whitespace'''
print('Preparing data...')
df.speech = df.speech+' . '
df.speech = df.speech.progress_apply(lambda x: x.replace('\n', ' '))
df.speech = df.speech.progress_apply(lambda x: x.replace(".", " . ")) #add space around dot
df.speech = df.speech.progress_apply(lambda x: re.sub('\s\s+' , ' ', x))  
df.speech = df.speech.progress_apply(lambda x: x.replace('. ', '. \n'))

print('Group by year...')
PERyear_df = df.copy().groupby(df.year)['speech'].progress_apply(''.join).reset_index() #concat sentences, each last sentence for each speech did not have dot so add one.

# Decade comparison before and after economic crisis
print('Selecting corpus before and after economic crisis...')
mask1 = (PERyear_df['year'] >= 1997) & (PERyear_df['year'] <= 2007)
corpus_before = '\n'.join([text for text in PERyear_df.loc[mask1].speech])

mask2 = (PERyear_df['year'] >= 2008) & (PERyear_df['year'] <= 2018)
corpus_after = '\n'.join([text for text in PERyear_df.loc[mask2].speech])

crisis_dichotomy_df = pd.DataFrame(data=[['1997_2007', corpus_before],
                                         ['2008_2018', corpus_after]],
                                   columns = ['period', 'speech'])

if isinstance(no_rows, str):
    if no_rows =='all':
        pass
    else:
        print('Undefined row number')
else:

    crisis_dichotomy_df.speech[0] = crisis_dichotomy_df.speech[0][:no_rows]
    crisis_dichotomy_df.speech[1] = crisis_dichotomy_df.speech[1][:no_rows]

crisis_dichotomy_df.head(2)

i=5
np.random.seed(i)
rn.seed(i)
my_seed = i

swifts_crisis_dichotomy_list=[]

print('Creating training texts...')

training_texts_dir = 'training_texts/'
if not os.path.exists(training_texts_dir):
    os.makedirs(training_texts_dir)

# Create compass text
compass_file_path = training_texts_dir+'crisis_dichotomy_compass.txt'
preprocessed_corpus = '\n'.join([decade_speech for decade_speech in crisis_dichotomy_df.speech.to_list()])

with open(compass_file_path, "w") as o:
    o.write(preprocessed_corpus)

periods = crisis_dichotomy_df.period.to_list()

for period, speech in tqdm(zip(crisis_dichotomy_df.period, crisis_dichotomy_df.speech)):
    with open(training_texts_dir+period+'.txt', "w") as o:
        o.write(speech) 

print('Training compass...')

aligner = CADE(size=vector_size, workers=1, diter= diter, siter = siter)
aligner.train_compass(compass_file_path, overwrite=True, save=True, seed=my_seed)

print('Training slices of two time periods')
             
m1 = aligner.train_slice(training_texts_dir+periods[0]+'.txt', save=True, seed = my_seed)
m2 = aligner.train_slice(training_texts_dir+periods[1]+'.txt', save=True, seed = my_seed)

print('Computing cosine similarity for each word of the common vocabulary of the models')
common_vocab = list(set(m1.wv.vocab).intersection(set(m2.wv.vocab)))

for word in tq.tqdm(common_vocab):
    cos_sim = compute_cosine_similarity(m1, m2, word)
    most_similar_words_period0 = m1.wv.most_similar(positive=[word], topn=10)
    most_similar_words_period1 = m2.wv.most_similar(positive=[word], topn=10)
    swifts_crisis_dichotomy_list.append([(periods[0], periods[1]), word, cos_sim, len(common_vocab), most_similar_words_period0, most_similar_words_period1])

swifts_crisis_dichotomy_df = pd.DataFrame(swifts_crisis_dichotomy_list, 
                                          columns = ['periods', 'word',
                                                     'semantic_similarity', 'common_voc_size',
                                                     'neighbors_t1', 'neighbors_t2'])

print(swifts_crisis_dichotomy_df.describe())

swifts_crisis_dichotomy_df = swifts_crisis_dichotomy_df.sort_values('semantic_similarity')

print('Words with the lowest cosine similarity / highest change')
print(swifts_crisis_dichotomy_df.head(10))

print('Words with the highest cosine similarity / lowest change')
print(swifts_crisis_dichotomy_df.tail(10))

# Take into account word frequency (at least 50 occurences in any time period)
freq_df_period0 = pd.read_csv('../out_files/freqs_for_semantic_shift_cleaned_data_period'+periods[0]+'.csv')
freq_df_period0 = freq_df_period0.sort_values('frequency', ascending=False).reset_index(drop=True)
freq_df_period0.columns = ['word', 'freq_period0', 'percent_period0']

freq_df_period1 = pd.read_csv('../out_files/freqs_for_semantic_shift_cleaned_data_period'+periods[1]+'.csv')
freq_df_period1 = freq_df_period1.sort_values('frequency', ascending=False).reset_index(drop=True)
freq_df_period1.columns = ['word', 'freq_period1', 'percent_period1']

# Bring top n results that have at least "at_least_in_any_decade" occurences in any of the two time periods
n = 100
at_least_in_any_decade = 50

df_topn = swifts_crisis_dichotomy_df.copy()

df_topn = df_topn[['word', 'semantic_similarity', 'neighbors_t1', 'neighbors_t2']]

df_topn = df_topn.merge(freq_df_period0, how='left', on='word')
df_topn = df_topn.merge(freq_df_period1, how='left', on='word')

df_topn['max_freq_of_any_decade'] = df_topn[['freq_period0', 'freq_period1']].max(axis=1)
df_topn.drop(['freq_period0', 'freq_period1', 'percent_period0', 'percent_period1'], axis=1, inplace=True) 
df_topn = df_topn.loc[df_topn.max_freq_of_any_decade>at_least_in_any_decade].head(n)

csv_path = '../out_files/semantic_shifts_dichotomy_crisis_compass_1997_2007_2008_2018_atleast50.csv'
df_topn.to_csv(csv_path, index=False)

print('Top changed words and their neighbors in the two time periods...')
n=10
neighbor_change = []
top_n_words = df_topn.word.head(n).to_list()
for word in top_n_words:
    most_similar_words_period0 = ',\n'.join([neighbor for neighbor, similarity in m1.wv.most_similar(positive=[word], topn=10)])
    most_similar_words_period1 = ',\n'.join([neighbor for neighbor, similarity in m2.wv.most_similar(positive=[word], topn=10)])
    neighbor_change.append([word, most_similar_words_period0, most_similar_words_period1, swifts_crisis_dichotomy_df.loc[(swifts_crisis_dichotomy_df.word==word), 'semantic_similarity']])
neighbor_change_df = pd.DataFrame(neighbor_change, columns = ['word', periods[0], periods[1], 'semantic_similarity'])
print(tabulate(neighbor_change_df, headers='keys', tablefmt='fancy_grid'))