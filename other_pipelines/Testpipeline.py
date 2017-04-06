# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:24:00 2017

@author: Laura
"""
#%%
#Load libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
import matplotlib.pyplot as plt
#from sklearn.ensemble import DecisionTreeClassifier as RF
#from sklearn.neighbors import KNeighborsClassifier as RF
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import confusion_matrix as  CM
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold as SKF

#%%
#Load Data (device ID, index, location and event ID are filtered out)
#
data_and_labels = pd.read_csv('events_combined.csv', usecols=[2,3,8,9,10,11,12], parse_dates = [4], na_values='XXX')
data_and_labels_no_nan = data_and_labels.dropna()
del data_and_labels
data = data_and_labels_no_nan.iloc[:,:6]
labels = data_and_labels_no_nan.iloc[:,6]
del data_and_labels_no_nan
one_hot_data = (pd.get_dummies(data.iloc[:3000,:], sparse=True)).as_matrix()
labels = labels.iloc[:3000]

lb = preprocessing.LabelBinarizer()
templabels=lb.fit_transform(labels) #also use to transform back
templabels = templabels*np.arange(templabels.shape[1])
numlabels = np.sum(templabels,axis=1)

#%%pre-processing (TODO: extract into modular function call)

#%%
#Create cross-validation splits (10-fold)
n = 10
skf1 = SKF(n_splits = n)

#%%
#Cross validating loop (double loop including hyperparameter optimalisation)
test_error = np.zeros(n)
confmat = np.zeros((templabels.shape[1],templabels.shape[1]))
i=0
#train_error = np.zeros(5)
for main_train_index, test_index in skf1.split(one_hot_data,numlabels):
    #skf2 = SKF(n_splits = n-1)
    #for val_train_index, val_index in skf2.split(one_hot_data[trainindex],labels[trainindex]):
     rf = RF(n_estimators=100)
     rf.fit(one_hot_data[main_train_index,:],numlabels[main_train_index])
     test_pred = rf.predict(one_hot_data[test_index,:])
     test_error[i] = ACC(numlabels[test_index],test_pred)
     confmat = confmat + CM(numlabels[test_index],test_pred)
     i=i+1
avg_error = np.mean(test_error)
#%%
#Upload promosing results
#save settings.