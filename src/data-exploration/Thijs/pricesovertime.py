# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:35:37 2017

@author: Thijs
"""

#%% Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(threshold=np.inf)

#%% Files

DATADIR = '../../../data/'

trainfile = DATADIR + 'train.csv'

trainset = pd.read_csv(trainfile)

#%% Preparing data

trainset['year'] = trainset['timestamp'].apply(lambda x: x[:4])
trainset['year-month'] = trainset['timestamp'].apply(lambda x: x[:4] + "-" + x[5:7])

#%% Counting the amount of entries by timestamp
plt.figure()
sns.countplot(x="year", data=trainset)

plt.figure(figsize=(12,8))
sns.countplot(x="year-month", data=trainset)
plt.xticks(rotation='vertical')

#%% Timestamps with median price

medianprice = trainset.groupby('year-month')['price_doc'].aggregate(np.median).reset_index().sort_values("year-month")

plt.figure(figsize=(12,8))
sns.barplot(x="year-month", y="price_doc", data=medianprice)
plt.xticks(rotation='vertical')
plt.ylabel('Median price')