# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:24:00 2017

@author: Laura
"""
#%%
#Load libraries
import pickle
import numpy as np
import pandas as pd
#from sklearn.ensemble import DecisionTreeClassifier as RF
#from sklearn.neighbors import KNeighborsClassifier as RF
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix as CM
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.utils.np_utils import probas_to_classes

#make vectorizer for tfidf processing
vectorizer = TfidfVectorizer(analyzer = "word")

#%%
#Code for fiting from generator by Chenglong Chen (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
            
def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

#%% Load general information and data
gendir = '../data/general/'
tfdir = '../data/Train_features/'
ttdir = '../data/Test_features/'
subdir = '../submission/'

if (os.path.isdir(gendir) != True):
    raise ValueError("gendir does not exist, please make sure '" + gendir + "' exists")
if (os.path.isdir(tfdir) != True):
    raise ValueError("tfdir does not exist, please make sure '" + tfdir + "' exists")
if (os.path.isdir(ttdir) != True):
    raise ValueError("ttdir does not exist, please make sure '" + ttdir + "' exists")
if (os.path.isdir(subdir) != True):
    raise ValueError("subdir does not exist, please make sure '" + subdir + "' exists")

try:
    train_labels = pickle.load(open(gendir + 'labels.p',"rb"))
    nclasses = pickle.load(open(gendir + 'nclasses.p',"rb"))
    label_encoding = pickle.load(open(gendir + 'label_encoding.p',"rb"))
    device_names = pickle.load(open(gendir + 'device_names.p',"rb"), encoding='latin1')
except IOError:
    print ("Please run `Feature_extraction.py` first. We do not have the correct Pickle files yet.")
    exit(-1)

#%%Load Features 

train_features = []
feature_names = []

tfcount = 0
for filename in os.listdir(tfdir):
    if (filename[-2:] != ".p"):
        continue
    if (filename[-2:] == ".p"):
        tfcount += 1

    train_features.append(pickle.load(open(tfdir + filename,"rb"), encoding='latin1'))
    feature_names.append(filename)

if tfcount == 0:
    print ("No pickle files found in training directory, please run Feature_extraction.py first")
    exit(-3)

test_features = []

ttcount = 0
for filename in os.listdir(ttdir):
    if (filename[-2:] != ".p"):
        continue
    if (filename[-2:] == ".p"):
        ttcount += 1
    test_features.append(pickle.load(open(ttdir + filename,"rb"), encoding='latin1'))

if ttcount == 0:
    print ("No pickle files found in test directory, please run Feature_extraction.py first")
    exit(-4)

    
#%% Combine Features

X_train_full = sp.sparse.hstack(train_features, format = 'csr')
X_test_full = sp.sparse.hstack(test_features, format = 'csr')

#%%
#Create cross-validation splits (10-fold)
n = 3
skf1 = SKF(n_splits = n)

#%%
#Cross validating loop
test_acc = np.zeros(n)
loss = np.zeros(n)

confmat = np.zeros((len(np.unique(train_labels)),len(np.unique(train_labels))))
i=0

input_shape = X_train_full.shape[1:]
model = Sequential()
model.add(Dense(100, input_shape=input_shape, init='uniform', activation='relu'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(50, init='uniform', activation='relu'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(12, init='uniform', activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Single loop: testing is done by submitting near-optimal results to Kaggle.
for train_index, test_index in skf1.split(np.zeros(len(train_labels)),train_labels):
     print ('start iteration ' + str(i))
     
     X_train = X_train_full[train_index]
     X_test = X_train_full[test_index]
     
     Y_train = train_labels[train_index]
     Y_test = train_labels[test_index]
     
     #-----------------------------------------------------------
     #Train classifier
     #-----------------------------------------------------------     
     
     fit = model.fit_generator(generator=batch_generator(X_train, Y_train, 100, True),
                               nb_epoch=5,
                               validation_data=(X_test.todense(), Y_test),
                               samples_per_epoch=X_train.shape[0])
     
     #-----------------------------------------------------------
     #Evaluate classifier
     #-----------------------------------------------------------
     test_proba = model.predict_generator(generator=batch_generatorp(X_test, 100, False),
                                          val_samples=X_train_full.shape[0])
     test_class = probas_to_classes(test_proba)
     test_acc[i] = ACC(Y_test,test_class)
     loss[i] = log_loss(train_labels[test_index],test_proba)
     confmat = confmat + CM(Y_test,test_class)
     i=i+1
     print ('end iteration ' + str(i))
     
     
avg_error = np.mean(test_acc)
avg_loss = np.mean(loss)
#%% Final prediction and csv saving
print ('starting final prediction')
fit = model.fit_generator(generator=batch_generator(X_train_full, train_labels, 100, True),
                          nb_epoch=15,
                          samples_per_epoch=X_train_full.shape[0])
pred_gen = model.predict_generator(generator=batch_generatorp(X_test_full, 100, False),
                                   val_samples=X_test_full.shape[0])
sub = pd.DataFrame(pred_gen, index = device_names, columns=label_encoding)

sub_file = 'testsub.csv'
sub.to_csv(subdir + sub_file, index=True)
print ('submission saved to ' + subdir + sub_file)