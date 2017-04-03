# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:24:00 2017

@author: Laura
"""
#%%
#Load libraries
import cPickle as pickle
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import DecisionTreeClassifier as RF
#from sklearn.neighbors import KNeighborsClassifier as RF
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix as  CM
#from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold as SKF
#from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp


#%% Load general information and data
gendir = './general/'

train_labels = pickle.load(open(gendir + 'labels.p',"rb"))
nclasses = pickle.load(open(gendir + 'nclasses.p',"rb"))
label_encoding = pickle.load(open(gendir + 'label_encoding.p',"rb"))
device_names = pickle.load(open(gendir + 'device_names.p',"rb"))

#%%Load Features 

train_features = []
feature_names = []

for filename in os.listdir('./Train_features/'):
        train_features.append(pickle.load(open('./Train_features/' + filename,"rb")))
        feature_names.append(filename)
    
test_features = []

for filename in os.listdir('./Test_features/'):
        test_features.append(pickle.load(open('./Test_features/' + filename,"rb")))
    
#%% Combine Features

X_train_full = sp.sparse.hstack(train_features, format = 'csr')
X_test_full = sp.sparse.hstack(test_features, format = 'csr')

#%%
#Create cross-validation splits (10-fold)
n = 10
skf1 = SKF(n_splits = n)

#%%
#Cross validating loop
test_acc = np.zeros(n)
loss = np.zeros(n)

confmat = np.zeros((len(np.unique(train_labels)),len(np.unique(train_labels))))
i=0

#Single loop: testing is done by submitting near-optimal results to Kaggle.
for train_index, test_index in skf1.split(np.zeros(len(train_labels)),train_labels):
     print 'start iteration ' + str(i)
     
     X_train = X_train_full[train_index]
     X_test = X_train_full[test_index]
     
     Y_train = train_labels[train_index]
     Y_test = train_labels[test_index]
     
     #-----------------------------------------------------------
     #Train classifier
     #-----------------------------------------------------------     
     
     #clf = RF(n_estimators=100, class_weight='balanced')
     clf = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs', class_weight='balanced')
     clf.fit(X_train,Y_train)
     
     #-----------------------------------------------------------
     #Evaluate classifier
     #-----------------------------------------------------------
     test_pred = clf.predict(X_test)
     test_proba = clf.predict_proba(X_test)
     test_acc[i] = ACC(Y_test,test_pred)
     loss[i] = log_loss(train_labels[test_index],test_proba)
     confmat = confmat + CM(Y_test,test_pred)
     i=i+1
     print 'end iteration ' + str(i)
     
     
avg_error = np.mean(test_acc)
avg_loss = np.mean(loss)
#%% Final prediction and csv saving
print 'starting final prediction'
#clf = RF(n_estimators=100, class_weight='balanced')
LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs', class_weight='balanced')
clf.fit(X_train_full,train_labels)
pred = pd.DataFrame(clf.predict_proba(X_test_full), index = device_names, columns=label_encoding)

pred.to_csv('testsub.csv',index=True)