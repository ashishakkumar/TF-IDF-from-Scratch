# -*- coding: utf-8 -*-
"""TF-IDF from Scratch & comparison with scikit-learn

## Consider the corpus as the list of documents
"""

corpus = [
    'this is the first document document.',
    'this is the first ',
   'this document is the second document.',
    'and this is the third one.',
    'is this the first document?',
 ]

"""## TF : Term Frequency

![Screenshot 2024-03-26 at 10.43.03 PM.png](attachment:983b87ec-ba37-4ff9-a1ca-2ff34590ce06.png)

- count(w,doc) : Number of times the word w appears in document doc
- |doc| : Number of words in the document

First let's remove puctuation from texts and  get all the distinct words from the documents :
"""

import re
def remove_punctuation(text):
  """Removes all punctuation from a string.

  Args:
    text: The string to remove punctuation from.

  Returns:
    A string with all punctuation removed.
  """

  pattern = re.compile(r'[^\w\s]')
  return pattern.sub(' ', text)

print(remove_punctuation(corpus[0]))

"""Now let's get all the unique words from the corpus"""

unique_words = set()
for i in corpus :
    only_txt = remove_punctuation(i)
    words = only_txt.split(' ')
    for j in range(len(words)-1) :
        words[j] = words[j].lower()
        unique_words.add(words[j])
unique_words_list = list(unique_words)
unique_words_list

"""#### Now this unique_words set is let's say the defining method for the (doc_id, word_index)
-> Word index for This : 0

-> Word index for document : 1
etc

We have two documents (TEXT lines) and let's say the index starts from 0
- Index for  'This is the first document.' : 0
- Index for  'This document is the second document.' : 1

#### Therefore count(0,1) means count of word "second" in the document "This is the first document"

### Calculating TF and storing them as a dictionary where keys are the tuples and values are the TF values
"""

def tf(corpus) :
    tf_dict = {}
    for i,doc in enumerate(corpus) :
        txt = remove_punctuation(doc).split(' ')[:-1]
        len_doc = len(txt)
        for j, word in enumerate(unique_words_list) :
                tf_dict[(i,j)] = txt.count(word)/len_doc
    return tf_dict

tf_dict = tf(corpus)

tf_dict

"""### Printing TF values

"""

def print_words_tf():
    for l,k in tf_dict.keys() :
        print(unique_words_list[k], tf_dict[(l,k)])
print_words_tf()

"""## IDF : Inverse Document Frequency

![Screenshot 2024-03-26 at 10.43.29 PM.png](attachment:0217a78a-1a52-4a1a-91f2-620681a9c9aa.png)

#### - Document Freqency df(w,D) : Number of documens that contains the target word w
#### - |D| : Total number of documents in the corpus
"""

import numpy as np
import math

np.log(math.e)

def IDF(word) :
    count = 0
    for doc in corpus :
        txt = remove_punctuation(doc).split(' ')[:-1]
        if word in txt :
            count+=1

    return np.log(len(corpus)/(count))

IDF_vals = []
idf_dict = {}
for k, word in enumerate(unique_words_list):
    idf_dict[k]= IDF(word)

idf_dict

"""### Printing the IDF values of the unique words"""

def print_words_idf():
    for i in range(len(unique_words_list)) :
        print(unique_words_list[i], idf_dict[i])
print_words_idf()

"""## TF-IDF

![Screenshot 2024-03-26 at 10.44.40 PM.png](attachment:45285184-b300-4257-a0dc-db581b67d1db.png)
"""

tf_idf_dict = {}
for wd_tuple in tf_dict.keys():
    tf_idf = tf_dict[wd_tuple]*idf_dict[wd_tuple[1]]
#     if tf_idf !=0 :
    tf_idf_dict[wd_tuple] = tf_idf

tf_idf_dict

"""## Comparison with scikit learn

1. TF calculation in scikit learn is done by just counting the occurence of words in the document
2. In IDF calculation, scikit-learn method add 1 after the log operation i.e. log(n/df) +1
"""

def scikit_tf(corpus):
    tf_dict = {}
    for i,doc in enumerate(corpus) :
        txt = remove_punctuation(doc).split(' ')[:-1]
        for j, word in enumerate(unique_words_list) :
                tf_dict[(i,j)] = txt.count(word) ## Just counting the word occurence
    return tf_dict
tf_dict_scikit = scikit_tf(corpus)
tf_dict_scikit

def scikit_IDF(word) :
    count = 0
    for doc in corpus :
        txt = remove_punctuation(doc).split(' ')[:-1]
        if word in txt :
            count+=1
    return np.log((len(corpus)/(count)))+1 ##Adding 1 after log operation

IDF_vals = []
idf_dict = {}
for k, word in enumerate(unique_words_list):
    idf_dict[k]= scikit_IDF(word)

idf_dict

"""### Multiplying the TF & IDF values"""

tf_idf_dict = {}
for wd_tuple in tf_dict_scikit.keys():
    tf_idf = tf_dict_scikit[wd_tuple]*idf_dict[wd_tuple[1]] ## TF * IDF
    tf_idf_dict[wd_tuple] = tf_idf

"""### Creating a numpy array for storing the term document matrix with TF-IDF values"""

import pandas as pd
def transformation(corpus, idf_dict,tf_idf_dict):
    my_df = pd.DataFrame()
    term_doc_array = np.zeros((len(corpus), len(idf_dict.keys())))
    for i in range(len(corpus)) :
        for j in range(len(idf_dict.keys())) :
            term_doc_array[i][j] = tf_idf_dict[(i,j)]
    return term_doc_array

term_doc_array = transformation(corpus, idf_dict,tf_idf_dict)

"""## Converting EVERYTHING into a class"""

class TF_IDF() :
    def __init__(self) :
        self.feature_names = None

    def fit_transform(self, corpus) :
        unique_words = set()
        for i in corpus :
            only_txt = remove_punctuation(i)
            words = only_txt.split(' ')
            for j in range(len(words)-1) :
                words[j] = words[j].lower()
                unique_words.add(words[j])
        self.feature_names = list(unique_words)
        # TF
        tf_dict = {}
        for i,doc in enumerate(corpus) :
            txt = remove_punctuation(doc).split(' ')[:-1]
            for j, word in enumerate(self.feature_names) :
                    tf_dict[(i,j)] = txt.count(word) ## Just counting the word occurence
        # IDF
        IDF_vals = []
        idf_dict = {}
        for k, word in enumerate(self.feature_names):
            count = 0
            for doc in corpus :
                txt = remove_punctuation(doc).split(' ')[:-1]
                if word in txt :
                    count+=1
            idf_dict[k]= np.log((len(corpus)/(count)))+1
        tf_idf_dict = {}
        for wd_tuple in tf_dict_scikit.keys():
            tf_idf = tf_dict_scikit[wd_tuple]*idf_dict[wd_tuple[1]] ## TF * IDF
            tf_idf_dict[wd_tuple] = tf_idf
        return tf_idf_dict


    def print_feature_names(self):
        return self.feature_names



tf_idf_instance = TF_IDF()
X = tf_idf_instance.fit_transform(corpus)
tf_idf_instance.print_feature_names()

print(X)

"""### 1. MY CODED OUTPUT"""

my_df = pd.DataFrame(term_doc_array, columns = unique_words_list)
my_df

"""### TF-IDF calculation based on Scikit-learn's TfidfVectorizer"""

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(smooth_idf = False, norm = None)
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()

"""### 2. SCIKIT LEARN'S OUTPUT"""

df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())
df

"""- SCIKIT LEARN'S OUTPUT and MY CODED OUTPUT matches
- Can try different versions of calculating TF and IDF on your choice

Important points :
- Need to convert entire document to lowercase before applying this task

## References :
1. https://www.youtube.com/watch?v=D3yL63aYNMQ&ab_channel=StanfordOnline
"""
