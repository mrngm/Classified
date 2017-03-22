# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:24:00 2017

@author: Laura
"""
#%%
#Load libraries
import cPickle as pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
#from sklearn.ensemble import DecisionTreeClassifier as RF
#from sklearn.neighbors import KNeighborsClassifier as RF
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix as  CM
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp

#make vectorizer for tfidf processing
vectorizer = TfidfVectorizer(analyzer = "word")
#%% logloss function
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
#%%
#Load Features, Data and labels.

apps_clean = pickle.load(open("stringdumpfinal.p","rb"))
#tfidf_features = pickle.load(open('termmat.p', 'rb'))
labels = pickle.load(open('labels.p', 'rb'))

#transform to numerical, one-hot, labels
lb = preprocessing.LabelBinarizer()
templabels=lb.fit_transform(labels) #also use to transform back
templabels = templabels*np.arange(templabels.shape[1])
numlabels = np.sum(templabels,axis=1)

#%%
#Create cross-validation splits (10-fold)
n = 10
skf1 = SKF(n_splits = n)

#%%
#Cross validating loop
test_acc = np.zeros(n)
loss = np.zeros(n)

confmat = np.zeros((templabels.shape[1],templabels.shape[1]))
i=0

#Single loop: testing is done by submitting near-optimal results to Kaggle.
for train_index, test_index in skf1.split(np.zeros(len(numlabels)),numlabels):
     print 'start new it'
     #-----------------------------------------------------------
     #Construct final feature vectors 
     #This is done in the loop because of tf-idf features. If features are only device
     #dependent, can be moved outside of the loop.
     #-----------------------------------------------------------
     #concatenate and constructfeatures (more can be added by using hstack())
     docs_train = [apps_clean[idx] for idx in train_index]
     docs_test = [apps_clean[idx] for idx in test_index]
          
     #X_train = hstack((vectorizer.fit_transform(apps_clean[train_index]),~~~))
     #X_test = hstack((vectorizer.transform(apps_clean[test_index]), ~~~))
     
     X_train = vectorizer.fit_transform(docs_train)
     X_test = vectorizer.transform(docs_test)
     
     Y_train = numlabels[train_index]
     Y_test = numlabels[test_index]
     
     #-----------------------------------------------------------
     #Train classifier
     #-----------------------------------------------------------     
     
     weights = {}
     for c in np.arange(len(np.unique(numlabels))):
         weights[c] = len(Y_train)/np.double(len(np.unique(numlabels))*np.bincount(Y_train))[c]
    
     rf = RF(n_estimators=1000, class_weight=weights)
     rf.fit(X_train,Y_train)
     
     #-----------------------------------------------------------
     #Evaluate classifier
     #-----------------------------------------------------------
     test_pred = rf.predict(X_test)
     test_proba = rf.predict_proba(X_test)
     test_acc[i] = ACC(Y_test,test_pred)
     loss[i] = log_loss(numlabels[test_index],test_proba)
     confmat = confmat + CM(Y_test,test_pred)
     i=i+1
     print 'end it'
     
     
avg_error = np.mean(test_acc)
avg_loss = np.mean(loss)
#%%
#Upload promising results
#save settings.