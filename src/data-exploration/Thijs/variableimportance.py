# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:59:29 2017

@author: Thijs
"""

#%% Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
import seaborn as sns
from sklearn import preprocessing
import operator
#import xgboost as xgb

np.set_printoptions(threshold=np.inf)               # print entire values

#%% XGBoost setup

import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb

#%% Files

DATADIR = '../../../data/'

trainfile = DATADIR + 'train.csv'

trainset = pd.read_csv(trainfile)

#%% XGBoost Feature Importance
# source: https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-sberbank

train_df = trainset

for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

try:
    model
except NameError:
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

#%% Feature Importance list and plot

sorted_scores = sorted(model.get_score().items(), key=operator.itemgetter(1), reverse=True)

for i in range(0,20):
    print (str(i+1) + ". " + str(sorted_scores[i][0]) + ": " + str(sorted_scores[i][1]))

# plot the important features #
fig, ax = plt.subplots()
xgb.plot_importance(model, max_num_features=20, height=0.8, ax=ax)
plt.show()

#%% full_sq

plt.figure()
plt.plot(trainset['full_sq'], trainset['price_doc'], "o",
         color="g", ms=5)
plt.show()

# Remove outlier
filtered = trainset[trainset['full_sq'] < 4000]

sns.jointplot(x="full_sq", y="price_doc", data=filtered,
              color="g", size=8, s=10)

#%% life_sq

plt.figure()
plt.plot(trainset['life_sq'], trainset['price_doc'], "o",
         color="g", ms=5)
plt.show()

# Remove outlier
filtered = trainset[trainset['life_sq'] < 7000]

sns.jointplot(x="life_sq", y="price_doc", data=filtered,
              color="g", size=8, s=10)

#%% floor

plt.figure(figsize=(12,8))
sns.countplot(x="floor", data=trainset)
plt.xticks(rotation='vertical')

floor_medianprice = trainset.groupby('floor')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(x="floor", y="price_doc", data=floor_medianprice)
plt.xticks(rotation='vertical')
plt.ylabel('Median price')

#%% max_floor

plt.figure(figsize=(12,8))
sns.countplot(x="max_floor", data=trainset)
plt.xticks(rotation='vertical')

max_floor_medianprice = trainset.groupby('max_floor')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(x="max_floor", y="price_doc", data=max_floor_medianprice)
plt.xticks(rotation='vertical')
plt.ylabel('Median price')

#%% build_year

buildyears = np.sort(trainset.build_year.unique()).astype(int)
print(buildyears)

# Remove outliers
filtered = trainset[~pd.isnull(trainset)]
filtered = filtered[(filtered.build_year > 1600)
    & (filtered.build_year != 4965)
    & (filtered.build_year != 20052009)]

sns.jointplot(x="build_year", y="price_doc", data=filtered,
              color="g", size=8, s=10)

#%% kitch_sq

plt.figure()
plt.plot(trainset['kitch_sq'], trainset['price_doc'], "o",
         color="g", ms=5)
plt.show()

# Get close-up of data
filtered = trainset[trainset['kitch_sq'] < 500]

sns.jointplot(x="kitch_sq", y="price_doc", data=filtered,
              color="g", size=8, s=10)

#%% state

plt.figure()
sns.countplot(x="state", data=trainset)

state_medianprice = trainset.groupby('state')['price_doc'].aggregate(np.median).reset_index()

plt.figure()
sns.barplot(x="state", y="price_doc", data=state_medianprice)
plt.ylabel('Median price')

#%% num_room

states = np.sort(trainset.state.unique())
print(states)

num_room_medianprice = trainset.groupby('num_room')['price_doc'].aggregate(np.median).reset_index()

plt.figure()
sns.barplot(x="num_room", y="price_doc", data=num_room_medianprice)
plt.ylabel('Median price')