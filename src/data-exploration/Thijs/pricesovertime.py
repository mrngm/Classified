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
trainset['month'] = trainset['timestamp'].apply(lambda x: x[5:7])
trainset['yearmonth'] = trainset['year'] + "-" + trainset['month']

#%% Counting the amount of entries by timestamp
plt.figure()
sns.countplot(x="year", data=trainset)

plt.figure(figsize=(12,8))
sns.countplot(x="yearmonth", data=trainset)
plt.xticks(rotation='vertical')

#%% Timestamps with median price

medianprice = trainset.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index().sort("yearmonth")

plt.figure(figsize=(12,8))
sns.barplot(x="yearmonth", y="price_doc", data=medianprice)
plt.xticks(rotation='vertical')