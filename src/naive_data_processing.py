# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:44:28 2017

@author: Roel
"""
# Dependencies and versions:
    # pandas 0.19.2
    # sklearn 0,18.1
    # Python 2.7
    # Included in the Anaconda 4.3.1 Python 2.7 64bit distribution for the Windows platform
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer

train_raw = pd.read_csv('../data/train.csv')
test_raw = pd.read_csv('../data/test.csv')

all_cols = list(train_raw.columns.values)

train_prices = train_raw['price_doc']

test_ids = test_raw['id']

non_num_cols = ['product_type', 'sub_area', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion','radiation_raion','railroad_terminal_raion', 'big_market_raion','nuclear_reactor_raion', 'detention_facility_raion', 'water_1line','big_road1_1line','railroad_1line','ecology']
redundant_cols = ['culture_objects_top_25', 'id', 'timestamp', 'price_doc']

train = train_raw[[col for col in all_cols if col not in redundant_cols and col not in non_num_cols]]
test = test_raw[[col for col in all_cols if col not in redundant_cols and col not in non_num_cols]]


le = LabelEncoder()
lb = LabelBinarizer()
enc = OneHotEncoder(sparse=False, handle_unknown = 'ignore')

for col in non_num_cols:
    
    train_part = lb.fit_transform(train_raw[col])
    test_part = lb.transform(test_raw[col])
    for i in range(train_part.shape[1]):
        
        train[col+'_'+str(i)] = train_part[:,i]
        test[col+'_'+str(i)] = test_part[:,i]

imp= Imputer(missing_values = "NaN", strategy = 'median')
train_no_nan = pd.DataFrame(imp.fit_transform(train),columns=train.columns.values)
test_no_nan = pd.DataFrame(imp.transform(test),columns=test.columns.values, index=test_ids)

train_no_nan.to_csv('../data/one-hot_median_filled_train.csv')
test_no_nan.to_csv('../data/one-hot_median_filled_test.csv')
train_prices.to_csv('../data/train_prices.csv')
