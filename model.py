from flask import Flask,render_template,request,flash
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
lema=WordNetLemmatizer()

data = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

corpus = []
punc = """:;""?!#@&.,"""
lema = WordNetLemmatizer()
for i in range(0, 1000):
       mess = [w for w in data.Review[i] if w not in punc]
       mess = "".join(mess)
       mess = mess.lower()
       all_stop_word = stopwords.words("english")
       all_stop_word.remove('not')
       all_stop_word.remove('no')
       all_stop_word.remove("didn't")
       all_stop_word.remove("won't")
       all_stop_word.remove("shan't")

       message = [lema.lemmatize(word) for word in mess.split() if word not in set(all_stop_word)]
       message = " ".join(message)
       corpus.append(message)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, -1].values
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X, y)
model = [cv, classifier]
with open("model_nlp.pkl", "wb") as f:
   pickle.dump(model, f)

