#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm import tqdm
import numpy as np
import datetime
import re
import os
tqdm.pandas()
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import tqdm.notebook as tq
from tabulate import tabulate
import itertools
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt

def compute_cosine_similarity(model1,model2,word):
    vector1 = model1.wv[word].reshape(1,-1)
    vector2 = model2.wv[word].reshape(1,-1)
    return(cosine_similarity(X=vector1, Y=vector2)[0][0])

def step_one_pairs(list_of_items):
    return [(list_of_items[i],list_of_items[i+1]) for i in range(len(list_of_items)-1)]


def train_word2vec(PERdecade_df, iteration, my_seed):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print('Start training')
    print(datetime.datetime.now())

    for decade, texts in tqdm(zip(PERdecade_df.decade, PERdecade_df.speech)):
        print(decade)
        model = Word2Vec(sentences=texts, size=300, window=4, min_count=20,
                         workers=1, seed=my_seed)
        model.save(models_dir + str(decade) + '_' + str(iteration) + ".mdl")

    print(datetime.datetime.now())


def collect_eligible_neighbors(model, plausible_neighbors, topn_neighbors):
    c = 0
    out = []
    for w, s in model.wv.most_similar(positive=[word], topn=10000):
        if w in plausible_neighbors:
            out.append(w)
            c += 1
        if c == topn_neighbors:
            break

    return (out)

def eligible_words(iteration, m1, m2, most_freq_1990, least_frequent_1990,
                   most_freq_2010, least_frequent_2010):
    ''' Collects words eligible for semantic shift computation from the
    intersection of the vocabularies that fulfill specific frequency thresholds'''
    m1_vocab = [key for key, value in m1.wv.vocab.items() if key != ' ']
    m2_vocab = [key for key, value in m2.wv.vocab.items() if key != ' ']

    intersection = set(m1_vocab).intersection(set(m2_vocab))
    most_freq = set(most_freq_1990 + most_freq_2010)
    least_freq = set(least_frequent_1990 + least_frequent_2010)

    # Clean words to search for usage change
    final_list = [w for w in intersection if
                  w not in most_freq and w not in least_freq and w != '@sw']
    print(len(final_list))

    return m1_vocab, m2_vocab, final_list

def eligible_neighbors(m1_vocab, m2_vocab, less_than_100_union):
    ''' Collects words that are eligible neighbors to the words studied for
    semantic shift eligible neighbors must be in both model vocabs and must
    appear more than 100 times in each corpus'''
    intersection_vocabs = list(set(m1_vocab).intersection(set(m2_vocab)))
    print("Vocabulary intersection: ", len(intersection_vocabs))
    plausible_neighbors = [w for w in intersection_vocabs if
                           w not in less_than_100_union and w != '']

    return plausible_neighbors

parser = ArgumentParser()
parser.add_argument("-r", "--run", type=int)
parser.add_argument("-i", "--iterations", type=int)
parser.add_argument("-z", "--size", type=int)
parser.add_argument("-n", "--rows", type=str)

args = vars(parser.parse_args())
run = args['run']
iterations = args['iterations']
vector_size = args['size']
no_rows = args['rows']

if no_rows.isdecimal():
    no_rows = int(no_rows)

df = pd.read_csv('../out_files/tell_all_cleaned.csv')
df = df[df['speech'].notna()]
df.sitting_date = pd.to_datetime(df.sitting_date, format="%d/%m/%Y")
df.speech.head(4)

#New column year
df['year'] = df['sitting_date'].dt.year
df['decade'] = (df['year']//10)*10
df = df[df.decade != 1980] # remove dates before 2000 to catch the three last decades
df = df[df.decade != 2020]# remove 2020s
df = df[df.decade != 2000]# remove 2000s

print(datetime.datetime.now())

print('Group by decades...')
PERdecade_df = df.groupby(df.decade)['speech'].progress_apply('.'.join).reset_index() # add missing dot from end of sentence
print('Tokenize...')
PERdecade_df.speech = PERdecade_df.speech.progress_apply(lambda x: [sent.split(' ') for sent in x.split('.')])
PERdecade_df.speech = PERdecade_df.speech.progress_apply(lambda x: [token for token in x if token!='' and token!=' '])

print(datetime.datetime.now())

if isinstance(no_rows, str):
    if no_rows =='all':
        pass
    else:
        print('Undefined row number')
else:

    PERdecade_df.speech[0] = PERdecade_df.speech[0][:no_rows]
    PERdecade_df.speech[1] = PERdecade_df.speech[1][:no_rows]

models_dir =  '../out_files/wordmodels/goldberg_decade/'
    
for i in range(iterations): 
    
    print('Training model...'+str(i))
    np.random.seed(i)
    random.seed(i)
    train_word2vec(PERdecade_df, i, i)  


#Collect words for semantic shift analysis that fulfill the thresholds

# Count frequency of words for each decade
df_freq_1990 = pd.read_csv('../out_files/freqs_for_semantic_shift_cleaned_data_decade1990.csv')
df_freq_2010 = pd.read_csv('../out_files/freqs_for_semantic_shift_cleaned_data_decade2010.csv')

print('1990')
df_freq_1990 = df_freq_1990[df_freq_1990.word != '@sw']
print(df_freq_1990.frequency.describe().apply(lambda x: format(x, 'f')))
# print(df_freq_1990.percentage.describe().apply(lambda x: format(x, 'f')))

print('2010')
df_freq_2010 = df_freq_2010[df_freq_2010.word != '@sw']
print(df_freq_2010.frequency.describe().apply(lambda x: format(x, 'f')))
# print(df_freq_2010.percentage.describe().apply(lambda x: format(x, 'f')))

# most frequent words at the top
df_freq_1990 = df_freq_1990.sort_values('frequency', ascending=False)
df_freq_2010 = df_freq_2010.sort_values('frequency', ascending=False)

# collect 200 most frequent words
most_freq_1990 = df_freq_1990.word.head(200).to_list()
most_freq_2010 = df_freq_2010.word.head(200).to_list()

# collect words with less than 200 frequency
least_frequent_1990 = df_freq_1990.loc[df_freq_1990.frequency<200].word.to_list()
least_frequent_2010 = df_freq_2010.loc[df_freq_2010.frequency<200].word.to_list()

# collect words with less than 100 occurrences 
less_than_100_1990_list = df_freq_1990.loc[(df_freq_1990.frequency<100), 'word'].to_list()
less_than_100_2010_list = df_freq_2010.loc[(df_freq_2010.frequency<100), 'word'].to_list()
less_than_100_union = set(less_than_100_1990_list+less_than_100_2010_list)

# Compute semantic shift
shifts_PERdecade_list=[]
error_list = []

topn_neighbors = 1000

for i in range(iterations):
    print('Iteration ', str(i))
    
    m1 = Word2Vec.load(models_dir+'1990_'+str(i)+'.mdl')
    m2 = Word2Vec.load(models_dir+'2010_'+str(i)+'.mdl')
    
    m1_vocab, m2_vocab, final_list = eligible_words(i, m1, m2, most_freq_1990, least_frequent_1990, most_freq_2010, least_frequent_2010)
    plausible_neighbors = eligible_neighbors(m1_vocab, m2_vocab, less_than_100_union)

    for word in tqdm(final_list):

        #union of neighbors in two points in time
        neighbors_t1 = collect_eligible_neighbors(m1, plausible_neighbors, topn_neighbors)
        neighbors_t2 = collect_eligible_neighbors(m2, plausible_neighbors, topn_neighbors)

        if len(neighbors_t1)<topn_neighbors or len(neighbors_t2)<topn_neighbors:
            error_list.append([word, len(neighbors_t1), len(neighbors_t2)])

        score = -len(set(neighbors_t1).intersection(set(neighbors_t2)))
        shifts_PERdecade_list.append([i, '1990-2010', word, score, neighbors_t1, neighbors_t2])
        
        

shifts_PERdecade_df = pd.DataFrame(shifts_PERdecade_list, columns = ['iteration', 'decade_pair', 'word',
                                                                 'semantic_similarity', 'neighbors_t1',
                                                                    'neighbors_t2'])
print(shifts_PERdecade_df.describe())

shifts_PERdecade_df = shifts_PERdecade_df.sort_values('semantic_similarity')

print('Words with the lowest score/ lowest change')
print(shifts_PERdecade_df.head(20))

print('Words with the highest score / highest change')
print(shifts_PERdecade_df.tail(20))

shifts_PERdecade_df.to_csv('../out_files/stability_goldberg_run'+str(run)+'_iterations'+str(
    iterations)+'_size'+str(vector_size)+'_rows'+str(no_rows)+'.csv', index=False)

# print('Are there any null values in the results? '+ str(shifts_PERdecade_df.isnull().values.any()))
# print('Are there any null words? null-words-dDataframe: ')
# null_df = shifts_PERdecade_df.loc[shifts_PERdecade_df['word'].isnull()]
# print(null_df)

topn_dict = {}
X = []
Y = []

k=[10,20,50,100,200,500,1000]

for n in k:
    
    for iteration in range(iterations):
        subdf = shifts_PERdecade_df.loc[(shifts_PERdecade_df.iteration==iteration)]
        subdf.sort_values('semantic_similarity', ascending=False).reset_index(drop=True)
        topn_dict[iteration] = subdf.head(n).word.to_list()
    
    topn_list_of_lists = [val for key, val in topn_dict.items()]
#     intersection = len(set(topn_dict[0]).intersection(set(topn_dict[1])))
    intersection = len(set(topn_list_of_lists[0]).intersection(*topn_list_of_lists))
    
    Y.append(intersection/n)
    X.append(n)
    
# print(X,Y)

fig = plt.figure(figsize=(15, 8))

fig.set_size_inches(20, 10)
plt.scatter(X,Y)
plt.plot(X,Y)
plt.gca().tick_params(axis='both', which='major', labelsize=15)

plt.xlabel('k', fontsize=18)
plt.ylabel('Intersection@k', fontsize=18)
plt.title('Stability for Goldberg', fontsize=20)

plt.savefig('../out_files/stability_goldberg_run'+str(run)+'_iterations'+str(
    iterations)+'_size'+str(vector_size)+'_rows'+str(no_rows)+'.png', dpi=200, bbox_inches='tight')