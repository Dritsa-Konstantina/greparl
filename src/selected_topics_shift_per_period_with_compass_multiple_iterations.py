import pandas as pd
import numpy as np
import datetime
import os
from cade.cade import CADE
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import random as rn
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

def compute_cosine_similarity(model1,model2,word):
    vector1 = model1.wv[word].reshape(1,-1)
    vector2 = model2.wv[word].reshape(1,-1)
    return(cosine_similarity(X=vector1, Y=vector2)[0][0])

def step_one_pairs(list_of_items):
    return [(list_of_items[i],list_of_items[i+1]) for i in range(len(list_of_items)-1)]

parser = ArgumentParser()
parser.add_argument("-s", "--seeds", nargs="+", type=int)

args = vars(parser.parse_args())
seeds = list(args['seeds'])

PERperiod_df = pd.read_csv('../out_files/PERperiod_df.csv')
training_texts_dir = 'training_texts/PERperiod/'

vouliwatch_topics = ['αγροτικη', 'αγροτικα', 'αγροτες', 'αναπτυξη', 'επενδυσεις',
                     'επενδυσεων', 'ασφαλιστικο', 'εργασια', 'εργασιακα',
                     'δικαιωματα', 'συμβαση', 'συλλογικη', 'μισθος', 'βασικος',
                     'ημιαπασχοληση', 'brain', 'drain', 'οαεδ', 'δικαιοσυνη',
                     'διαφανεια', 'υιοθεσια', 'υιοθεσιας', 'εθνικη', 'αμυνα',
                     'ενοπλες', 'εξωτερικη', 'διεθνεις', 'ναυτιλια','νησια',
                     'υδροδοτηση','αλιεια', 'οικονομια', 'αφορολογητο',
                     'αποκεντρωση', 'φπα', 'επιχειρησεις', 'φορολογικα',
                     'φορολογια', 'παιδεια', 'ερευνα', 'προσχολικη', 'ασυλο',
                     'ασυλου', 'πανεπιστημιακο', 'δευτεροβαθμια', 'πρωτοβαθμια',
                     'ανωτατη', 'δημοσια', 'ιδιωτικη', 'σχολες', 'περιβαλλον',
                     'ενεργεια', 'συστημα', 'πολιτισμος', 'αθλητισμος', 'πολιτισμο',
                     'αθλητισμου', 'προστασια', 'αστυνομια', 'κοκ', 'αστυνομικη',
                     'αστυνομικοι', 'αστυνομικων', 'προσφυγικο', 'μεταναστευτικο',
                     'προσφυγες', 'μεταναστες', 'τουρισμος', 'τουριστικης',
                     'τουρισμου', 'τουρισμο', 'υγεια', 'προνοια', 'επιδοματα',
                     'επιδομα', 'επιδοματων', 'υποδομες', 'μεταφορες', 'αναπλαση',
                     'μμμ', 'μειωση', 'αυξηση', 'συνταξη', 'μακεδονικο', 'μακεδονιας',
                     'μακεδονια', 'προσληψεις', 'απολυσεις', 'εκας', 'οσε',
                     'συγκοινωνιες', 'νομοσχεδιο', 'θρησκεια', 'θρησκειας',
                     'θρησκευτικο', 'θρησκευτικων', 'γυναικα', 'γυναικας',
                     'ανδρας', 'ανδρα', 'αντρας', 'αντρα', 'ομοφυλα',
                     'ομοφυλοφιλοι', 'ομοφιλοφιλων', 'εοπυυ', 'τουρκια', 'τουρκιας',
                     'εκκλησιας', 'εκκλησια', 'κριση', 'κρισης', 'αυθαιρεσια',
                    ]

shifts_pp_list = []

periods = sorted(PERperiod_df.period.to_list())
period_pairs = step_one_pairs(periods)

for i in seeds:

    print('Iteration ', str(i))
    print('Iteration started at ', str(datetime.datetime.now()))

    np.random.seed(i)
    rn.seed(i)
    my_seed = i
    
    print('Creating directory for compass files of concatenated period pairs...')
    compass_dir = training_texts_dir+'topics/seed_'+str(i)+'/'
    if not os.path.exists(compass_dir):
        os.makedirs(compass_dir)
    print(compass_dir)

    for pair in period_pairs:
        period_1, period_2 = str(pair[0]),str(pair[1])
        print(pair)
        compass_file_path = compass_dir+str(period_1)+'.'+str(period_2)+'.txt'
        with open(compass_file_path, "w") as o:
            o.write(open(training_texts_dir+period_1+'.txt').read()+"\n"+open(training_texts_dir+period_2+'.txt').read())

        aligner = CADE(size=300, workers=1, opath=compass_dir)
        aligner.train_compass(compass_file_path, overwrite=True, save=False, seed=my_seed)
        m1 = aligner.train_slice(training_texts_dir+period_1+'.txt', save=False, seed=my_seed)
        m2 = aligner.train_slice(training_texts_dir+period_2+'.txt', save=False, seed=my_seed)

        common_vocab = list(set(m1.wv.vocab).intersection(set(m2.wv.vocab)))

        for topic in vouliwatch_topics:
            if (topic in m1.wv.vocab) and (topic in m2.wv.vocab):
                cos_sim = compute_cosine_similarity(m1, m2, topic)
                most_similar_words_period0 = m1.wv.most_similar(positive=[topic], topn=20)
                most_similar_words_period1 = m2.wv.most_similar(positive=[topic], topn=20)
                shifts_pp_list.append([i, pair, topic, cos_sim, len(common_vocab),
                                       most_similar_words_period0, most_similar_words_period1])
        
        os.remove(compass_file_path)
        
    print('Iteration ended at ', str(datetime.datetime.now()))


shifts_pp_df = pd.DataFrame(shifts_pp_list, columns = ['iteration', 'period_pair', 'word', 
                                                     'semantic_similarity', 'common_voc_size',
                                                        'neighbors_t1',
                                                        'neighbors_t2'])

shifts_pp_df = shifts_pp_df.sort_values('semantic_similarity')
shifts_pp_df.to_csv('../out_files/selected_topics_shift_per_period_compass_50iterations_seeds_'+'_'.join([str(i) for i in seeds])+'.csv',
                    index=False)