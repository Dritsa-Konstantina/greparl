#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import re
import os
from cade.cade import CADE
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import random as rn
import seaborn as sns
import random
import ast
from collections import defaultdict
import matplotlib.pyplot as plt

def compute_cosine_similarity(model1,model2,word):
    vector1 = model1.wv[word].reshape(1,-1)
    vector2 = model2.wv[word].reshape(1,-1)
    return(cosine_similarity(X=vector1, Y=vector2)[0][0])

def step_one_pairs(list_of_items):
    return [(list_of_items[i],list_of_items[i+1]) for i in range(len(list_of_items)-1)]


def translate_word(word):
    lookup = {'αγροτικη': 'agricultural', 'αγροτικα': 'agricultural',
              'αγροτες': 'farmers',
              'αναπτυξη': 'growth', 'επενδυσεις': 'investements',
              'επενδυσεων': 'investements',
              'ασφαλιστικο': 'insurance', 'εργασια': 'labor',
              'εργασιακα': 'labor', 'δικαιωματα': 'rights',
              'συμβαση': 'contract', 'συλλογικη': 'collective',
              'μισθος': 'salary', 'βασικος': 'minimum',
              'ημιαπασχοληση': 'part-time', 'brain': 'brain', 'drain': 'drain',
              'οαεδ': 'OAED',
              'δικαιοσυνη': 'justice', 'διαφανεια': 'transparency',
              'υιοθεσια': 'adoption', 'υιοθεσιας': 'adoption',
              'εθνικη': 'national', 'αμυνα': 'defence', 'ενοπλες': 'armed',
              'εξωτερικη': 'foreign', 'διεθνεις': 'international',
              'ναυτιλια': 'shipping', 'νησια': 'islands',
              'υδροδοτηση': 'water_supply', 'αλιεια': 'fishing',
              'οικονομια': 'economy', 'αφορολογητο': 'tax-exempt',
              'αποκεντρωση': 'decentralization', 'φπα': 'VAT',
              'επιχειρησεις': 'businesses', 'φορολογικα': 'tax',
              'φορολογια': 'taxation',
              'παιδεια': 'education', 'ερευνα': 'research',
              'προσχολικη': 'preschool', 'ασυλο': 'asylum', 'ασυλου': 'asylum',
              'πανεπιστημιακο': 'university', 'δευτεροβαθμια': 'secondary',
              'πρωτοβαθμια': 'primary', 'ανωτατη': 'higher',
              'δημοσια': 'public', 'ιδιωτικη': 'private',
              'σχολες': 'schools/faculties',
              'περιβαλλον': 'environment', 'ενεργεια': 'energy',
              'συστημα': 'system',
              'πολιτισμος': 'culture', 'αθλητισμος': 'sports',
              'πολιτισμο': 'culture', 'αθλητισμου': 'sports',
              'προστασια': 'protection', 'αστυνομια': 'police',
              'κοκ': 'traffic_code', 'αστυνομικη': 'police',
              'αστυνομικοι': 'police', 'αστυνομικων': 'police',
              'προσφυγικο': 'refugee', 'μεταναστευτικο': 'migratory',
              'προσφυγες': 'refugees', 'μεταναστες': 'immigrants',
              'τουρισμος': 'tourism', 'τουριστικης': 'tourism',
              'τουρισμου': 'tourism', 'τουρισμο': 'tourism',
              'υγεια': 'heatlh', 'προνοια': 'welfare', 'επιδοματα': 'subsidies',
              'επιδομα': 'subsidy', 'επιδοματων': 'subsidies',
              'υποδομες': 'infrastructure', 'μεταφορες': 'transportation',
              'αναπλαση': 'remodeling', 'μμμ': 'public_transport ',
              'μειωση': 'reduction', 'αυξηση': 'raise', 'συνταξη': 'retirement',
              'μακεδονια': 'macedonia', 'μακεδονιας': 'macedonia',
              'μακεδονικο': 'macedonian', 'προσληψεις': 'hirings',
              'απολυσεις': 'redundancies', 'εκας': 'EKAS', 'οσε': 'OSE',
              'συγκοινωνιες': 'transportation', 'νομοσχεδιο': 'bill',
              'θρησκεια': 'religion', 'θρησκειας': 'religion',
              'θρησκευτικο': 'religious', 'θρησκευτικου': 'religious',
              'θρησκευτικων': 'religious', 'γυναικα': 'woman',
              'γυναικας': 'woman', 'ανδρας': 'man', 'αντρας': 'man',
              'αντρα': 'man', 'ανδρα': 'man', 'ομοφυλα': 'same-sex',
              'ομοφυλοφιλοι': 'homosexuals', 'ομοφιλοφιλων': 'homosexuals',
              'εοπυυ': 'EOPPY', 'τουρκια': 'Turkey', 'τουρκιας': 'Turkey',
              'εκκλησιας': 'church', 'εκκλησια': 'church', 'κριση': 'crisis',
              'κρισης': 'crisis', 'αυθαιρεσια': 'arbitrariness'}

    if word in lookup.keys():
        return lookup[word]
    return word


def periods_to_dates(period_pair):
    period_dict = {5: [1989],
                   6: [1989, 1990],
                   7: [1990, 1993],
                   8: [1993, 1996],
                   9: [1996, 2000],
                   10: [2000, 2004],
                   11: [2004, 2007],
                   12: [2007, 2009],
                   13: [2009, 2012],
                   14: [2012],
                   15: [2012, 2014],
                   16: [2015],
                   17: [2015, 2019],
                   18: [2019, 2020]}

    super_periods = {7: [5, 6, 7], 15: [14, 15], 17: [16, 17]}

    new_periods = []
    new_dates = []

    for period in period_pair:
        if period in super_periods.keys():
            new_periods.append(
                '-'.join([str(p) for p in super_periods[period]]))  # 5-6-7
            new_dates.append((period_dict[super_periods[period][0]][0],
                              period_dict[super_periods[period][-1]][-1]))
        else:
            new_periods.append(period)
            new_dates.append(period_dict[period])

    #     new_period_pair = 'vs '.join([str(p) for p in new_periods])
    #     new_date_range = '('+'-'.join([str(d1) for d1 in new_dates[0]])+',\n'+'-'.join([str(d2) for d2 in new_dates[1]])+ ')'
    out = str(new_periods[0]) + ' (' + '-'.join(
        [str(d1) for d1 in new_dates[0]]) + ') &\n' + (
              str(new_periods[1])) + ' (' + '-'.join(
        [str(d2) for d2 in new_dates[1]]) + ')'
    #     new_period_pair = 'vs '.join([str(p) for p in new_periods])
    #     new_date_range = '('+'-'.join([str(d1) for d1 in new_dates[0]])+',\n'+'-'.join([str(d2) for d2 in new_dates[1]])+ ')'
    #     out = new_period_pair+'\n'+new_date_range

    return (out)


def plot_swift(vouliwatch_topics, shifts_pp_df):
    most_changed_topics = []

    plt.figure(figsize=(20, 10))

    for topic in vouliwatch_topics:
        topic_subdf = shifts_pp_df.loc[(shifts_pp_df.word == topic)]

        if topic_subdf.shape[0] > 0:

            topic_subdf = topic_subdf.sort_values('period_pair')
            period_pairs = topic_subdf.period_pair.to_list()
            X = [(pair[0] + pair[1]) / 2 for pair in period_pairs]
            Y = topic_subdf.semantic_similarity.to_list()
            #             print(Y)

            # plot only those that go below 0.5
            if (not all(sim >= 0.65 for sim in Y)) and topic not in [
                'προσχολικη', 'κοκ',
                'οσε', 'αθλητισμος', 'αθλητισμου',
                'υδροδοτηση',  # 'ανδρας', 'αντρας', 'αντρα',
                'οαεδ', 'αλιεια', 'εργασιακα',
                'εκκλησιας']:
                most_changed_topics.append(topic)
                plt.scatter(X, Y)
                plt.plot(X, Y, label=translate_word(topic))  # connect dots

                # annotate with topic name the lower point
                min_sim_index = Y.index(min(Y))

                #                 add_x = (random.uniform(-0.5,0.5))
                #                 add_y = (random.uniform(-0.045,-0.02))
                add_y = -0.06

                plt.annotate(text=translate_word(topic),
                             xy=(X[min_sim_index], Y[min_sim_index]),
                             xytext=(X[min_sim_index] + add_y,
                                     Y[min_sim_index] + add_y),
                             arrowprops=dict(arrowstyle="->"),
                             size=16
                             )

    middle_periods = [(pair[0] + pair[1]) / 2 for pair in
                      step_one_pairs(periods)]
    x_labels = [periods_to_dates(pair) for pair in step_one_pairs(periods)]

    plt.xticks(middle_periods, x_labels, rotation=70, fontsize=16)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, bbox_to_anchor=(1, 0.5))  # ncol=2,
    plt.rcParams['axes.labelsize'] = 19
    plt.ylim([0, 1])
    plt.title('Usage change of selected topics through time', fontsize=20)
    plt.ylabel('Cosine Similarity', fontsize=20)
    plt.xlabel('Pairs of Parliamentary Periods', fontsize=20)
    plt.savefig('../out_files/selected_topics_shift_per_period_compass.png',
                dpi=200, bbox_inches='tight')
    plt.show()
    return (most_changed_topics)

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
print(datetime.datetime.now())
#concat sentences, each last sentence for each speech did not have dot so add one.
PERperiod_df = df.groupby(df.period)['speech'].apply(''.join).reset_index()

# Create file per year
training_texts_dir = 'training_texts/PERperiod/'
if not os.path.exists(training_texts_dir):
    os.makedirs(training_texts_dir)

for period, speech in zip(PERperiod_df.period, PERperiod_df.speech):
    with open(training_texts_dir+str(period)+'.txt', "w") as o:
        o.write(speech)

vouliwatch_topics = ['αγροτικη', 'αγροτικα', 'αγροτες', 
          'αναπτυξη', 'επενδυσεις', 'επενδυσεων',
          'ασφαλιστικο', 'εργασια', 'εργασιακα', 'δικαιωματα', 'συμβαση', 'συλλογικη', 'μισθος', 'βασικος', 'ημιαπασχοληση', 'brain', 'drain', 'οαεδ',
          'δικαιοσυνη', 'διαφανεια', 'υιοθεσια', 'υιοθεσιας',
          'εθνικη', 'αμυνα', 'ενοπλες',
          'εξωτερικη', 'διεθνεις',
          'ναυτιλια','νησια','υδροδοτηση','αλιεια',
          'οικονομια', 'αφορολογητο', 'αποκεντρωση', 'φπα', 'επιχειρησεις', 'φορολογικα', 'φορολογια',
          'παιδεια', 'ερευνα', 'προσχολικη', 'ασυλο', 'ασυλου', 'πανεπιστημιακο', 'δευτεροβαθμια', 'πρωτοβαθμια', 'ανωτατη', 'δημοσια', 'ιδιωτικη', 'σχολες',
          'περιβαλλον', 'ενεργεια', 
          'συστημα',
          'πολιτισμος', 'αθλητισμος', 'πολιτισμο', 'αθλητισμου',
          'προστασια', 'αστυνομια', 'κοκ', 'αστυνομικη', 'αστυνομικοι', 'αστυνομικων',
          'προσφυγικο', 'μεταναστευτικο', 'προσφυγες', 'μεταναστες',
          'τουρισμος', 'τουριστικης', 'τουρισμου', 'τουρισμο',
          'υγεια', 'προνοια', 'επιδοματα', 'επιδομα', 'επιδοματων',
          'υποδομες', 'μεταφορες', 'αναπλαση', 'μμμ',
          'μειωση', 'αυξηση', 'συνταξη', 'μακεδονικο', 'μακεδονιας', 'μακεδονια', 'προσληψεις', 'απολυσεις', 'εκας', 'οσε', 'συγκοινωνιες', 'νομοσχεδιο',
                     'θρησκεια', 'θρησκειας', 'θρησκευτικο', 'θρησκευτικων', 'γυναικα', 'γυναικας', 'ανδρας', 'ανδρα', 'αντρας', 'αντρα', 'ομοφυλα', 'ομοφυλοφιλοι', 'ομοφιλοφιλων', 'εοπυυ', 'τουρκια', 'τουρκιας', 'εκκλησιας', 'εκκλησια', 'κριση', 'κρισης', 'αυθαιρεσια',
                     
         ]

#sort dataframe by period
periods = sorted(PERperiod_df.period.to_list())
period_pairs = step_one_pairs(periods)

np.random.seed(5)
rn.seed(5)
my_seed = 5
shifts_pp_list = []

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

    for topic in vouliwatch_topics:
        if (topic in m1.wv.vocab) and (topic in m2.wv.vocab):
            cos_sim = compute_cosine_similarity(m1, m2, topic)
            most_similar_words_period0 = m1.wv.most_similar(positive=[topic], topn=20)
            most_similar_words_period1 = m2.wv.most_similar(positive=[topic], topn=20)
            shifts_pp_list.append([pair, topic, cos_sim, len(common_vocab),
                                   most_similar_words_period0, most_similar_words_period1])

shifts_pp_df = pd.DataFrame(shifts_pp_list, columns = ['period_pair', 'word', 
                                                     'semantic_similarity', 'common_voc_size',
                                                        'neighbors_t1',
                                                        'neighbors_t2'])

shifts_pp_df = shifts_pp_df.sort_values('semantic_similarity')

print('Words with the lowest cosine similarity / highest change')
print(shifts_pp_df.head(20))

print('Words with the highest cosine similarity / lowest change')
print(shifts_pp_df.tail(20))
shifts_pp_df.to_csv('../out_files/selected_topics_shift_per_period_compass.csv', 
                           index=False)

# Present results

shifts_pp_df = pd.read_csv('../out_files/selected_topics_shift_per_period_compass.csv',
                          converters={'period_pair':ast.literal_eval})
# print(set(shifts_pp_df.word.to_list()))
most_changed_topics = plot_swift(vouliwatch_topics, shifts_pp_df)
