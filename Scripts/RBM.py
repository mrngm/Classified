# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:22:59 2017

@author: Gebruiker
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#%%
print 'importing'
df_train = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/macro.csv", parse_dates=['timestamp'])
#%%
print 'cleaning'
df_train = df_train[df_train.build_year != 20052009]
#%%
train = df_train.isnull().sum()
macro = df_macro.isnull().sum()
test = df_test.isnull().sum()
#%%
train_NaN = train[train != 0]
macro_NaN = macro[macro != 0]
test_NaN = test[test != 0]
#%%
