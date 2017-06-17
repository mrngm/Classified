# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 11:18:58 2017

@author: Gebruiker
"""

import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import exp, expm1
from math import log
#%%
print 'import data'
train = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/one-hot_median_filled_train.csv')
test = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/one-hot_median_filled_test.csv')
target = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/train_prices.csv')

train = train[1:]
train = train.drop('Unnamed: 0', 1)
test = test.drop('Unnamed: 0', 1)
#%%
X_train, X_test, y_train, y_test = train_test_split (train, target, test_size = .5)

clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)

test_pred = clf.predict(X_test)
print test_pred 

#%%
rms = sqrt(mean_squared_error(log(y_test), log(test_pred)))
