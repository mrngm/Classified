# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:00:24 2017

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
from sklearn.preprocessing import OneHotEncoder

Test = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Noise Eliminated Universal Files/Universal_1.csv', index_col = 0, dtype={'gender': object, 'group': object, 'phone_brand': object, 'device_model': object, 'category': str})
Groep = pd.read_excel('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/label_groep.xlsx')
Test.columns
del Test['group']
del Test['gender']
del Test['age']
del Test['app_id']
del Test['longitude']
del Test['latitude']
del Test['timestamp']
del Test['phone_brand']
del Test['device_model']
del Test['event_id']
del Test['is_installed']
del Test['is_active']
Test.head(3)
Test = Test.sort(['device_id'], ascending= [1])


Binary = pd.merge(left=Test,right=Groep, how='left', left_on='label_id', right_on='label_id')
Binary = Binary.sort(['device_id'], ascending= [1])
print Binary
Exp = Binary.head(100)
print Exp
Industry = Exp.groupby('Group').count()
del Industry['label_id']
del Industry['device_id']
print Industry
Columns = Industry.T
print Columns

Bank = Columns.Bank
Industry = Columns.Industry
Other = Columns.Other

"""
Animation = 7
Bank = 111
Board Games = 10
Books = 24
Comic = 10
Country = 35
Game = 116
Industry = 59
Other = 217
Products = 151
School = 37
Shopping = 6
Social = 17
Sports = 18
Standard = 52
Travel = 59
"""

#1:
unique_labels = []
for all rows:
	label = get_label_from_row()
	if label not in unique_labels:
		unique_labels.append(label)
#2:
label_counts = np.zeros((nr_unique_devices,len(unique_labels)))
for all devices: (enumerate to get row index for label_counts)
	device_data = get_data_for_device()
	for row in device_data:
		for l in unique_labels: (enumerate to get column index for label_counts)
			if l == row.label:
				label_counts[R_index,C_index] = label_counts[R_index,C_index]+1

#3:
label_counts_weighted = label_counts where every element in a row is divided by the sum of that row











