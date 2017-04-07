# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:11:19 2017

@author: Gebruiker
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV

Universal = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Noise Eliminated Universal Files/Universal.csv', index_col = 0, parse_dates=['timestamp'], infer_datetime_format=True)
Universal.columns
Universal.drop('is_installed', inplace=True, axis=1)
Universal.drop('is_active', inplace=True, axis=1)
Universal.drop('device_id', inplace=True, axis=1)
Universal.drop('gender', inplace=True, axis=1)
Universal.drop('age', inplace=True, axis=1)
Universal.drop('event_id', inplace=True, axis=1)
Universal.drop('app_id', inplace=True, axis=1)
Universal.drop('label_id', inplace=True, axis=1)
Universal.drop('timestamp', inplace=True, axis=1)
v = DV()
qualitative_features = ['phone_brand','device_model','category']
X_qual = v.fit_transform(Universal[qualitative_features].to_dict('records'))

P = X_qual.toarray()