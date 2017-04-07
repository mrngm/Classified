# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 19:34:00 2017

@author: Gebruiker
"""
from __future__ import division
"""
Programming Preparation
"""

import numpy as np
import pandas as pd
import datetime as dt
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

Phone_gender_geo = pd.read_csv('D:/School/School/Master/Jaar 1/Machine Learning in Practice/Competition/Data/Merge/Sub/Phone_Gender_Geo.csv', index_col = 0, dtype={'gender': object, 'group': str, 'phone_brand': str, 'device_model': str})
Phone_gender_geo = Phone_gender_geo[pd.notnull(Phone_gender_geo['gender'])]

Events = pd.read_csv('D:/School/School/Master/Jaar 1/Machine Learning in Practice/Competition/Data/Mobile Data/app_events.csv')
col_list = list(Events)
col_list[0]
col_list[1]
col_list[2]
col_list[3]
col_list[4]
col_list[5]
col_list[0], col_list[3] = col_list[3], col_list[0]
Events.columns = col_list

Universal = pd.merge(left=Phone_gender_geo,right=Events, left_on='event_id', right_on='event_id')
Universal.shape
print Universal
Universal.to_csv('Universal.csv')

Universal = pd.read_csv('D:/School/School/Master/Jaar 1/Machine Learning in Practice/Competition/Scripts/Universal.csv', index_col = 0, dtype={'gender': object, 'group': str, 'phone_brand': str, 'device_model': str})
Applabels = pd.read_csv('D:/School/School/Master/Jaar 1/Machine Learning in Practice/Competition/Data/Mobile Data/app_labels.csv')
Universal_1 = pd.merge(left=Universal,right=Applabels, left_on='app_id', right_on='app_id')
Universal_1.shape
print Universal_1
Universal_1.to_csv('Universal_1.csv')

Universal_1 = pd.read_csv('D:/School/School/Master/Jaar 1/Machine Learning in Practice/Competition/Scripts/Universal_1.csv', index_col = 0, dtype={'gender': object, 'group': str, 'phone_brand': str, 'device_model': str})
Labelscat = pd.read_csv('D:/School/School/Master/Jaar 1/Machine Learning in Practice/Competition/Data/Mobile Data/label_categories.csv')
Universal_2 = pd.merge(left=Universal_1,right=Labelscat, left_on='label_id', right_on='label_id')
Universal_2.shape
print Universal_2
Universal_2.to_csv('Universal_2.csv')