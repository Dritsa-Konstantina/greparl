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

def compute_cosine_similarity(model1,model2,word):
    vector1 = model1.wv[word].reshape(1,-1)
    vector2 = model2.wv[word].reshape(1,-1)
    return(cosine_similarity(X=vector1, Y=vector2)[0][0])

def step_one_pairs(list_of_items):
    return [(list_of_items[i],list_of_items[i+1]) for i in range(len(list_of_items)-1)]

tqdm.pandas()

parser = ArgumentParser()
parser.add_argument("-r", "--run", type=int)
parser.add_argument("-i", "--iterations", type=int)
parser.add_argument("-d", "--diter", type=int)
parser.add_argument("-s", "--siter", type=int)
parser.add_argument("-z", "--size", type=int)
parser.add_argument("-n", "--rows", type=str)

args = vars(parser.parse_args())
run = args['run']
iterations = args['iterations']
diter = args['diter']
siter = args['siter']
vector_size = args['size']
no_rows = args['rows']

if no_rows.isdecimal():
    no_rows = int(no_rows)

df = pd.read_csv('../out_files/tell_all_cleaned.csv')
df = df[df['speech'].notna()]
df.sitting_date = pd.to_datetime(df.sitting_date, format="%d/%m/%Y") 

#New column year
df['year'] = df['sitting_date'].dt.year
df['decade'] = (df['year']//10)*10
df = df[df.decade != 1980] # remove dates before 2000 to catch the three last decades
df = df[df.decade != 2020]# remove 2020s
df = df[df.decade != 2000]# remove 2000s

df.speech = df.speech.progress_apply(lambda x: x.replace(".", " . ")) #add space around dot
df.speech = df.speech+' . '

#concat sentences, each last sentence for each speech did not have dot so add one.
print('Preparing data...')
df.speech = df.speech.progress_apply(lambda x: x.replace('\n', ' '))
df.speech = df.speech.progress_apply(lambda x: re.sub('\s\s+' , ' ', x)) 
df.speech = df.speech.progress_apply(lambda x: x.replace('. ', '.\n'))
 
print('Group by decades...')
print(datetime.datetime.now())

#concat sentences, each last sentence for each speech did not have dot so add one.
PERdecade_df = df.groupby(df.decade)['speech'].progress_apply(''.join).reset_index()

# Select subset of the data for short training and testing
if isinstance(no_rows, str):
    if no_rows =='all':
        pass
    else:
        print('Undefined row number')
else:

    PERdecade_df.speech[0] = PERdecade_df.speech[0][:no_rows]
    PERdecade_df.speech[1] = PERdecade_df.speech[1][:no_rows]

shifts_PERdecade_list=[]

for i in range(iterations):
    print('********************************************************')
    print('Repeat No ', str(i))

    np.random.seed(i)
    rn.seed(i)
    my_seed = i

    print(datetime.datetime.now())

    print('Creating training texts...')
    training_texts_dir = '../out_files/training_texts/compass_stability/'
    if not os.path.exists(training_texts_dir):
        os.makedirs(training_texts_dir)

    for decade, speech in tqdm(zip(PERdecade_df.decade, PERdecade_df.speech)):
        with open(training_texts_dir+str(decade)+'_'+str(i)+'.txt', "w", encoding='utf-8') as o:
            o.write(speech)  

    PERdecade_df = PERdecade_df.sort_values(by='decade')
    decades = sorted(PERdecade_df.decade.to_list())
    decade_pairs = step_one_pairs(decades)

    print('Training aligned models...')
    print(datetime.datetime.now())

    for pair in tqdm(decade_pairs):
        decade_1, decade_2 = str(pair[0]),str(pair[1])
        print(pair)
        compass_file_path = training_texts_dir+str(decade_1)+'.'+str(decade_2)+'.txt'
        with open(compass_file_path, "w", encoding='utf-8') as o:
            o.write(open(training_texts_dir+decade_1+'_'+str(i)+'.txt', 
                         encoding='utf-8').read()+"\n"+open(training_texts_dir+decade_2+'_'+str(i)+'.txt', 
                                                            encoding='utf-8').read())

        aligner = CADE(size=vector_size, workers=1, diter= diter, siter = siter, #siter=1, diter=9
                      )
        aligner.train_compass(compass_file_path, overwrite=True, save=True, 
                              seed=my_seed
                             )
        m1 = aligner.train_slice(training_texts_dir+decade_1+'_'+str(i)+'.txt', save=True, 
                                 seed = my_seed
                                )
        m2 = aligner.train_slice(training_texts_dir+decade_2+'_'+str(i)+'.txt', save=True, 
                                 seed = my_seed
                                )

        common_vocab = list(set(m1.wv.vocab).intersection(set(m2.wv.vocab)))
        print(len(common_vocab))

        for word in common_vocab:

            cos_sim = compute_cosine_similarity(m1, m2, word)
            most_similar_words_period0 = m1.wv.most_similar(positive=[word], topn=10)
            most_similar_words_period1 = m2.wv.most_similar(positive=[word], topn=10)
            shifts_PERdecade_list.append([i, pair, word, cos_sim,
                                          len(common_vocab),
                                          most_similar_words_period0,
                                          most_similar_words_period1])

shifts_PERdecade_df = pd.DataFrame(shifts_PERdecade_list, columns = [
    'iteration', 'decade_pair', 'word', 'semantic_similarity',
    'common_voc_size', 'top10neighbors_1st_decade',
    'top10neighbors_2nd_decade'])

print(shifts_PERdecade_df.describe())

shifts_PERdecade_df = shifts_PERdecade_df.sort_values('semantic_similarity')

print('Words with the lowest cosine similarity / highest change')
print(shifts_PERdecade_df.head(20))

print('Words with the highest cosine similarity / lowest change')
print(shifts_PERdecade_df.tail(20))
shifts_PERdecade_df.to_csv('../out_files/stability_compass_run'+str(run)+'_iterations'+str(
    iterations)+'_diter'+str(diter)+'_siter'+str(siter)+'_size'+str(vector_size)+'_rows'+str(no_rows)+'.csv', 
                           index=False)

# Plot results
topn_dict = {}
X = []
Y = []

k=[10,20,50,100,200,500,1000]

for n in k:

    for iteration in range(iterations):
        subdf = shifts_PERdecade_df.loc[(shifts_PERdecade_df.iteration==iteration)]
        subdf = subdf.sort_values('semantic_similarity', ascending=False).reset_index(drop=True)
        topn_dict[iteration] = subdf.head(n).word.to_list()
#     print(topn_dict)
    topn_list_of_lists = [val for key, val in topn_dict.items()]
    intersection = len(set(topn_list_of_lists[0]).intersection(*topn_list_of_lists))

    Y.append(intersection/n)
    X.append(n)

# print(X,Y)

fig = plt.figure(figsize=(15, 8))

fig.set_size_inches(20, 10)
plt.scatter(X,Y)
plt.plot(X,Y)
plt.gca().tick_params(axis='both', which='major', labelsize=15)
plt.ylim(0,1.)
plt.xlabel('k', fontsize=18)
plt.ylabel('Intersection@k', fontsize=18)
plt.title('Stability for Compass', fontsize=20)

plt.savefig('../out_files/stability_compass_run'+str(run)+'_iterations'+str(iterations)+'_diter'+str(
    diter)+'_siter'+str(siter)+'_size'+str(vector_size)+'_rows'+str(no_rows)+'.png', dpi=200,  bbox_inches='tight')