import re
import pandas as pd
import json
import math

def text_formatting(text):

    text = re.sub("['’`΄‘́̈]",'', text)
    text = re.sub('\t+' , ' ', text)
    text = text.lstrip()
    text = text.rstrip()
    text = re.sub('\s\s+' , ' ', text)
    text = re.sub('\s*(-|–)\s*' , '-', text) #fix dashes
    text = text.lower()
    text = text.translate(str.maketrans('άέόώήίϊΐiύϋΰ','αεοωηιιιιυυυ')) #remove accents
    # convert english characters to greek
    text = text.translate(str.maketrans('akebyolruxtvhmnz','ακεβυολρυχτνημνζ'))

    return text

with open('../out_files/wiki_data/male_name_cases_populated.json') as \
        males_file, open('../out_files/wiki_data/female_name_cases_populated'
                         '.json') as females_file:

    male_names = list(json.load(males_file).keys())
    male_names.extend(['διακος', 'τσετιν', 'σπυροπανος', 'σπυριδωνας', 'τερενς',
                     'αιχαν', 'χουσειν', 'πυρρος', 'γκαληπ', 'μπηρολ', 'φιντιας',
                     'τραιανος', 'αχμετ', 'αθηναιος', 'φρανς', 'τζαννης',
                     'ροβερτος', 'μουσταφα', 'κλεων', 'παρισης', 'παυσανιας',
                      'μεχμετ', 'αμετ', 'μπουρχαν', 'πανουργιας', 'γιανης',
                      'ιλχαν', 'πυθαγορας', 'φραγκλινος', 'ισμαηλ', 'θαλασσινος'])
    male_dict = {text_formatting(m):"male" for m in male_names}

    female_names = list(json.load(females_file).keys())
    female_names.extend(['ελεωνορα', 'κρινιω', 'ιωαννετα', 'σουλτανα', 'ηρω',
                       'συλβα', 'χρυσουλα', 'ελισσαβετ', 'βιργινια', 'ροδουλα',
                        'καλλιοπη', 'γεσθημανη', 'φερονικη', 'χρυση', 'ολυμπια',
                        'καλλιοπη', 'μαριορη', 'παναγιου'
                         ])
    female_dict = {text_formatting(f):"female" for f in female_names}

    unisex = list(set(male_dict.keys()).intersection(female_dict.keys()))

    all_names = female_dict.copy()
    all_names.update(male_dict)

gender_df = pd.DataFrame.from_dict(all_names, orient='index', columns = ['gender'])
gender_df.first_name = gender_df.index.copy()
gender_df.reset_index(inplace=True)
gender_df.columns = ['first_name', 'gender']

members_df = pd.read_csv('../out_files/parl_members_activity_1989onwards.csv')

members_df['first_name'] = members_df.member_name.apply(lambda x: x.split(' '
                                                                          '')[2].split('-')[0])

if set(unisex).intersection(set(members_df['first_name'].to_list())):
    print('There will be conflict for the following unisex names. Please '
          'correct manually.')
    print(set(unisex).intersection(set(members_df['first_name'].to_list())))

df_with_gender = pd.merge(members_df, gender_df, left_on='first_name',
                          right_on='first_name', how='left')

# Specific correction
df_with_gender.loc[(df_with_gender.member_name =='πουλου λεωνιδα παναγιου ('
                                                 'γιωτα)'), 'gender'] = 'female'

df_with_gender.drop('first_name', axis=1, inplace=True)

if df_with_gender[df_with_gender['gender'].isnull()].shape[0]!=0:
    print(
        'Gender not found for following entries. Adjust script accordingly or '
        'proceed to manual correction.')
    print(df_with_gender[df_with_gender['gender'].isnull()])

df_with_gender.to_csv(
    '../out_files_trials/parl_members_activity_1989onwards_with_gender.csv',
    header=True, index=False, encoding='utf-8')