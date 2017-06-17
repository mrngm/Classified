# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:04:55 2017

@author: Gebruiker
"""

import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing
from sklearn.base import TransformerMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import scipy as sp
#%%
# Declare paths
data_folder_loc = "C:/Users/Gebruiker/data/"
train_loc = data_folder_loc + "train.csv"
train_subset_loc = data_folder_loc + "train_subset.csv"
validation_loc = data_folder_loc + "validation.csv"
val_prices_loc = data_folder_loc + "val_prices.csv"
#%%
train = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/Validatieset/train_2014.csv', parse_dates=['timestamp'])
x_test = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/Validatieset/test_2015.csv', parse_dates=['timestamp'])
y_test = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/Validatieset/val_prices.csv', usecols=['5813453'])

x_test = x_test.ix[1:]

train = train.assign(
    date_year = lambda d: d['timestamp'].dt.year,
    date_month = lambda d: d['timestamp'].dt.month,
    date_week = lambda d: d['timestamp'].dt.week,
    date_day = lambda d: d['timestamp'].dt.day,
    date_hour = lambda d: d['timestamp'].dt.hour,
    date_minute = lambda d: d['timestamp'].dt.minute,
    date_second = lambda d: d['timestamp'].dt.second,
    date_weekday = lambda d: d['timestamp'].dt.weekday,
)

x_test = x_test.assign(
    date_year = lambda d: d['timestamp'].dt.year,
    date_month = lambda d: d['timestamp'].dt.month,
    date_week = lambda d: d['timestamp'].dt.week,
    date_day = lambda d: d['timestamp'].dt.day,
    date_hour = lambda d: d['timestamp'].dt.hour,
    date_minute = lambda d: d['timestamp'].dt.minute,
    date_second = lambda d: d['timestamp'].dt.second,
    date_weekday = lambda d: d['timestamp'].dt.weekday,
)

train = train.drop('timestamp', 1)
x_test = x_test.drop('timestamp', 1)
#%%
#clean data
bad_index = train[train.life_sq > train.full_sq].index
train.ix[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
bad_index = train[train.life_sq < 5].index
train.ix[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.ix[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
train.ix[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index
train.ix[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index 
train.ix[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.ix[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.ix[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.ix[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.ix[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.ix[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.ix[bad_index, "floor"] = np.NaN
train.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.ix[bad_index, "state"] = np.NaN

# brings error down a lot by removing extreme price per sqm
train.loc[train.full_sq == 0, 'full_sq'] = 50
train = train[train.price_doc/train.full_sq <= 600000]
train = train[train.price_doc/train.full_sq >= 10000]

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)
train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
train['room_size'] = train['life_sq'] / train['num_room'].astype(float)

#%%
train = train[train.date_year == 2015]
#%%
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[c].values)) 
        train[c] = lbl.transform(list(train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        
#Split on instances before(<) and in(>=) 2015
imp= Imputer(missing_values = "NaN", strategy = 'median')
train = pd.DataFrame(imp.fit_transform(train),columns=train.columns.values)
x_test = pd.DataFrame(imp.fit_transform(x_test),columns=x_test.columns.values)

#%%

#%%
# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
linear.score(X_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
score= linear.predict(X_test)
#%%
# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model.fit(X_train, y_train)
model.score(X_train, y_train)
#Predict Output
score= model.predict(X_test)
#%%
# Create KNeighbors classifier object model 
model = KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
# Train the model using the training sets and check score
model.fit(X_train, y_train)
#Predict Output
score= model.predict(X_test)
#%%
# Create Random Forest object
model= RandomForestClassifier()
# Train the model using the training sets and check score
model.fit(X_train, y_train)
#Predict Output
score= model.predict(X_test)
#%%
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
score = est.predict(X_test)

#%%
def eval_rmsle(y_test,score):
    return np.sqrt(np.mean(np.square(np.subtract(np.log(np.add(score,1)),np.log(np.add(y_test,1))))))

RMSLE = np.sqrt(np.mean(np.square(np.subtract(np.log(np.add(score,1)),np.log(np.add(y_test,1))))))
print RMSLE