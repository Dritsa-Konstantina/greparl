#!/usr/bin/env python
# coding: utf-8

from gensim.models import Word2Vec
from tqdm import tqdm
import tqdm.notebook as tq
import pandas as pd
import nltk
import numpy as np
import gensim
from tqdm import tqdm
tqdm.pandas()
import itertools
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
nltk.download('punkt')
import os
import glob
import shutil
import re
import random as rn
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from argparse import ArgumentParser

def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
    (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary
    in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    base_embed.init_sims()
    other_embed.init_sims()

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed,
                                                              other_embed,
                                                              words=words)

    # get the embedding matrices
    # base_vecs = in_base_embed.syn0norm
    base_vecs = in_base_embed.wv.vectors_norm
    # other_vecs = in_other_embed.syn0norm
    other_vecs = in_other_embed.wv.vectors_norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm) by "ortho"
    # other_embed.syn0norm = other_embed.syn0 = (other_embed.syn0norm).dot(ortho)
    other_embed.wv.vectors_norm = other_embed.wv.vectors = (
        other_embed.wv.vectors_norm).dot(ortho)
    return other_embed


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.vocab.keys())
    vocab_m2 = set(m2.wv.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count,
                      reverse=True)

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        old_arr = m.wv.vectors_norm
        new_arr = np.array([old_arr[index] for index in indices])
        # m.syn0norm = m.syn0 = new_arr
        m.wv.vectors_norm = m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.wv.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index,
                                                           count=old_vocab_obj.count)
        m.wv.vocab = new_vocab

    return (m1, m2)

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

print('Preparing data...')

#New column year
df.sitting_date = pd.to_datetime(df.sitting_date, format="%d/%m/%Y") 
df['year'] = df['sitting_date'].dt.year
df['decade'] = (df['year']//10)*10
df = df[df.decade != 1980] # remove dates before 2000 to catch the three last decades
df = df[df.decade != 2020]# remove 2020s
df = df[df.decade != 2000]# remove 2000s

print('Group by decades...')
print(datetime.datetime.now())
df.speech = df.speech.progress_apply(lambda x: x.replace('\n', ' '))
#concat sentences, each last sentence for each speech did not have dot so add one.
PERdecade_df = df.groupby(df.decade)['speech'].progress_apply('.'.join).reset_index()
PERdecade_df.speech = PERdecade_df.speech.progress_apply(lambda x: [sent.split(' ') for sent in x.split('.')])
PERdecade_df.speech = PERdecade_df.speech.progress_apply(lambda x: [token for token in x if token!='' and token!=' '])

if isinstance(no_rows, str):
    if no_rows =='all':
        pass
    else:
        print('Undefined row number')
else:
    PERdecade_df.speech[0] = PERdecade_df.speech[0][:no_rows]
    PERdecade_df.speech[1] = PERdecade_df.speech[1][:no_rows]

shifts_PERdecade_list=[]

models_dir =  '../out_files/wordmodels/procrustes_stability/'

if not os.path.exists(models_dir):
    print('Creating models directory...')
    os.makedirs(models_dir)
    
#Create aligned folder
align_dest_dir = models_dir +'aligned/'
if not os.path.exists(align_dest_dir):
    print('Creating aligned models directory...')
    os.makedirs(align_dest_dir)

PERdecade_df.sort_values(by='decade')
decades = sorted(PERdecade_df.decade.to_list())
decade_pairs = step_one_pairs(decades)
    
for i in range(iterations):
    
    np.random.seed(i)
    rn.seed(i)
    my_seed=i
        
    print('********************************************************')
    print('Repeat No ', str(i))
    
    print(datetime.datetime.now())
    
    print('Training models for each decade...')

    for decade, texts in tqdm(zip(PERdecade_df.decade, PERdecade_df.speech)):
        print(decade)
        model = Word2Vec(sentences=texts, size=vector_size, window=5,
                         min_count=20, workers=1, seed=my_seed)
        model.save(models_dir+str(decade)+'_'+str(i)+ ".mdl")

    print(datetime.datetime.now())

    print('Aligning models...')
    print(datetime.datetime.now())

    for file in glob.glob(models_dir +str(decades[0])+'_'+str(i)+'.mdl*'):
        shutil.copy(file, align_dest_dir)
    
    m_t0 = Word2Vec.load(models_dir+str(decades[0])+'_'+str(i)+'.mdl')
    m_t1 = Word2Vec.load(models_dir+str(decades[1])+'_'+str(i)+'.mdl')
    m_t1_aligned = smart_procrustes_align_gensim(m_t0, m_t1)
    m_t1_aligned.save(align_dest_dir+str(decades[1])+'_'+str(i)+'.mdl')

    m1 = m_t0
    m2 = m_t1_aligned
    
    common_vocab = list(set(m1.wv.vocab).intersection(set(m2.wv.vocab)))
    print('Common vocab length... ', str(len(common_vocab)))
    print('Computing word similarity between decades....')

    for word in tq.tqdm(common_vocab):

        cos_sim = compute_cosine_similarity(m1, m2, word)
        most_similar_words_period0 = m1.wv.most_similar(positive=[word], topn=10)
        most_similar_words_period1 = m2.wv.most_similar(positive=[word], topn=10)
        shifts_PERdecade_list.append([i, decade_pairs[0], word, cos_sim,
                                      len(common_vocab),
                                      most_similar_words_period0, most_similar_words_period1])

    print(datetime.datetime.now())

shifts_PERdecade_df = pd.DataFrame(shifts_PERdecade_list, columns = ['iteration', 'decade_pair', 'word', 
                                                                 'semantic_similarity', 'common_voc_size',
                                                                    'top10neighbors_1st_decade',
                                                                    'top10neighbors_2nd_decade'])
print(shifts_PERdecade_df.describe())

shifts_PERdecade_df = shifts_PERdecade_df.sort_values('semantic_similarity')

print('Words with the lowest cosine similarity / highest change')
print(shifts_PERdecade_df.head(20))

print('Words with the highest cosine similarity / lowest change')
print(shifts_PERdecade_df.tail(20))

shifts_PERdecade_df.to_csv('../out_files/stability_procrustes_run'+str(run)+'_iterations'+str(iterations)+
                           '_size'+str(vector_size)+'_rows'+str(no_rows)+'.csv', 
                           index=False)

topn_dict = {}
X = []
Y = []

k=[10,20,50,100,200,500,1000]

for n in k:
    
    for iteration in range(iterations):
        subdf = shifts_PERdecade_df.loc[(shifts_PERdecade_df.iteration==iteration)]
        subdf = subdf.sort_values('semantic_similarity', ascending=False).reset_index(drop=True)
        topn_dict[iteration] = subdf.head(n).word.to_list()
    
    topn_list_of_lists = [val for key, val in topn_dict.items()]

    intersection = len(set(topn_list_of_lists[0]).intersection(*topn_list_of_lists))

    Y.append(intersection/n)
    X.append(n)

fig = plt.figure(figsize=(15, 8))

fig.set_size_inches(20, 10)
plt.scatter(X,Y)
plt.plot(X,Y)
plt.gca().tick_params(axis='both', which='major', labelsize=15)
plt.ylim(0,1.)
plt.xlabel('k', fontsize=18)
plt.ylabel('Intersection@k', fontsize=18)
plt.title('Stability for Procrustes', fontsize=20)

plt.savefig('../out_files/stability_procrustes_run'+str(run)+'_iterations'+str(iterations)+
            +'_size'+str(vector_size)+'_rows'+str(no_rows)+'.png', dpi=200,  bbox_inches='tight')

