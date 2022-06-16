#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import datetime
import re
import os
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from cade.cade import CADE
import random as rn
import seaborn as sns
import matplotlib.pyplot as plt
from decimal import Decimal
from collections import defaultdict
import random

def periods_to_dates(period_pair):
    period_dict_merged = {5: '1989',
                          6: '1989-1990',
                          7: '1990-1993',
                          8: '1993-1996',
                          9: '1996-2000',
                          10: '2000-2004',
                          11: '2004-2007',
                          12: '2007-2009',
                          13: '2009-2012',
                          14: '2012',
                          15: '2012-2014',
                          16: '2015',
                          17: '2015-2019',
                          18: '2019-2020'}

    out = str(period_pair) + '\n' + period_dict_merged[period_pair[0]] + ' & ' + \
          period_dict_merged[period_pair[1]]
    return (out)

def compute_cosine_similarity(model1,model2,word):
    vector1 = model1.wv[word].reshape(1,-1)
    vector2 = model2.wv[word].reshape(1,-1)
    return(cosine_similarity(X=vector1, Y=vector2)[0][0])

def step_one_pairs(list_of_items):
    return [(list_of_items[i],list_of_items[i+1]) for i in range(len(list_of_items)-1)]

df = pd.read_csv('../out_files/tell_all_cleaned.csv')
df = df[df['speech'].notna()]
df.sitting_date = pd.to_datetime(df.sitting_date, format="%d/%m/%Y")
df.speech = df.speech.apply(lambda x: x.replace(".", " . ")) #add space around dot
df.speech = df.speech+' . '

#concat sentences, each last sentence for each speech did not have dot so add one.
print('Preparing data...')
'''
cade tool uses gensim.models.word2vec.LineSentence() to iterate over the training corpus
gensim.models.word2vec.LineSentence() takes as input 
a file that contains sentences: one line = one sentence.
Words must be already preprocessed and separated by whitespace'''
df.speech = df.speech.apply(lambda x: x.replace('\n', ' '))
df.speech = df.speech.apply(lambda x: re.sub('\s\s+' , ' ', x)) 
df.speech = df.speech.apply(lambda x: x.replace('. ', '.\n'))

df = df.rename(columns={'parliamentary_period': 'period'})

# Adjust period names, merge small periods with larger and remove words in order to easily sort later on
df.period = df.period.apply(lambda x: x.replace(' review 9',''))
df.period = df.period.apply(lambda x: x.replace('period ',''))
df.period = df.period.astype(int)
df.loc[(df.period==5), 'period'] = 7
df.loc[(df.period==6), 'period'] = 7
df.loc[(df.period==14), 'period'] = 15
df.loc[(df.period==16), 'period'] = 17

print('Group by periods...')
#concat sentences, each last sentence for each speech did not have dot so add one.
PERperiod_df = df.groupby(df.period)['speech'].apply(''.join).reset_index()

training_texts_dir = 'training_texts/PERperiod/'
if not os.path.exists(training_texts_dir):
    os.makedirs(training_texts_dir)

for period, speech in zip(PERperiod_df.period, PERperiod_df.speech):
    with open(training_texts_dir+str(period)+'.txt', "w") as o:
        o.write(speech)

np.random.seed(5)
rn.seed(5)
my_seed = 5

shifts_pp_list=[]

periods = sorted(PERperiod_df.period.to_list())
period_pairs = step_one_pairs(periods)

for pair in period_pairs:
    period_1, period_2 = str(pair[0]),str(pair[1])
    print(pair)
    compass_file_path = training_texts_dir+str(period_1)+'.'+str(period_2)+'.txt'
    with open(compass_file_path, "w") as o:
        o.write(open(training_texts_dir+period_1+'.txt').read()+"\n"+open(training_texts_dir+period_2+'.txt').read())
    
    aligner = CADE(size=300, workers=1)
    aligner.train_compass(compass_file_path, overwrite=True, save=True, seed=my_seed)
    m1 = aligner.train_slice(training_texts_dir+period_1+'.txt', save=True, seed=my_seed)
    m2 = aligner.train_slice(training_texts_dir+period_2+'.txt', save=True, seed=my_seed)

    common_vocab = list(set(m1.wv.vocab).intersection(set(m2.wv.vocab)))

    for word in common_vocab:
        
        if '@' in word:

            cos_sim = compute_cosine_similarity(m1, m2, word)
            most_similar_words_period0 = m1.wv.most_similar(positive=[word], topn=20)
            most_similar_words_period1 = m2.wv.most_similar(positive=[word], topn=20)
            shifts_pp_list.append([pair, word, cos_sim, len(common_vocab), most_similar_words_period0, most_similar_words_period1])


shifts_pp_df = pd.DataFrame(shifts_pp_list, columns = ['period_pair', 'word',
                                                       'semantic_similarity', 'common_voc_size',
                                                       'neighbors_t1', 'neighbors_t2'])
print(shifts_pp_df.describe())

shifts_pp_df = shifts_pp_df.sort_values('semantic_similarity')

print('Words with the lowest cosine similarity / highest change')
print(shifts_pp_df.head(20))

print('Words with the highest cosine similarity / lowest change')
print(shifts_pp_df.tail(20))
shifts_pp_df.to_csv('../out_files/semantic_shifts_party_embeddings_per_period_merged_compass.csv', index=False)

parties_all = ['@μερα25', '@νδ', '@δησυ', '@συριζα', '@ελληνικη_λυση', '@πολαν', 
    '@ανεξαρτητοι_δημοκρατικοι_βουλευτες', 
    '@ανελ','@δηανα','@κιναλ','@δηκκι',
    '@συνασπισμος','@πασοκ','@κκε',
    '@λαος','@', 'χα','@οε','@λαε', 
    '@ποταμι','@εκ','@δημαρ']

parties = ['@νδ', '@συριζα', '@συνασπισμος', '@πασοκ', '@κκε', '@χα']

plt.figure(figsize=(20, 10)) 
colors = sns.color_palette('colorblind').as_hex()
# plt.xlim([5, 18])
plt.ylim([0, 1.])

for party in parties:

#     if party in shifts_PERperiod_df.word.to_list():
    party_subdf = shifts_pp_df.loc[(shifts_pp_df.word==party)]
    
    if party_subdf.shape[0]>1:
        
        party_subdf = party_subdf.sort_values('period_pair')
        X_pairs = party_subdf.period_pair.to_list()
        X = [(x[0]+x[1])/2 for x in X_pairs]
        Y = party_subdf.semantic_similarity.to_list()
        if len(X)<=4:
            pass
        
        else:

            plt.scatter(X,Y)
            plt.plot(X,Y, label=party)
            min_index = Y.index(min(Y))

            plt.annotate(party,
                fontsize=19, 
                 xy=(X[min_index], Y[min_index]), 
                 xytext=(X[min_index], Y[min_index]),
                )
            
            
middle_pairs = [(pair[0]+pair[1])/2 for pair in step_one_pairs(periods)]
x_labels = [periods_to_dates(pair) for pair in step_one_pairs(periods)] 
plt.xticks(middle_pairs,x_labels, rotation=70)

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.labelsize'] = 25

plt.xticks()
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend()
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize=20)

# We change the fontsize of minor ticks label 
ax.tick_params(axis='both', which='major', labelsize=15)
    
# plt.xlim([0, 1])
plt.ylim([0.35, 1.])
plt.ylabel('Cosine Similarity', fontsize=20)
plt.xlabel('Periods', fontsize=20)
plt.title('Semantic similarity for Party Embeddings per Period Pairs', fontsize=20)
plt.savefig('../out_files/semantic_shifts_party_embeddings_per_period_merged_compass.png', dpi=200, bbox_inches='tight')
plt.show()