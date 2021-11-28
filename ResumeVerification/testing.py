# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 17:24:46 2020

@author: ningesh
"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
listofclasses=['ENFJ',
 'ENFP',
 'ENTJ',
 'ENTP',
 'ESFJ',
 'ESFP',
 'ESTP',
 'INFJ',
 'INFP',
 'INTJ',
 'INTP',
 'ISFJ',
 'ISFP',
 'ISTJ',
 'ISTP']
filename='naive_bayes_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
Corpus = pd.read_csv(r"processed_data.csv",encoding='latin-1',nrows=10,error_bad_lines=False)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])

predictions_RF = loaded_model.predict(Tfidf_vect.transform(['hi hello']))
print(listofclasses[predictions_RF[0]-1])