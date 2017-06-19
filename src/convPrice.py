import numpy as np
import pandas as pd
prices = pd.read_csv('../data/train_prices.csv')
train = pd.read_csv('../data/one-hot_median_filled_train.csv')

# Declare paths
data_folder_loc = "../data/"
train_loc = data_folder_loc + "train.csv"
train_subset_loc = data_folder_loc + "train_2014.csv"
validation_loc = data_folder_loc + "test_2015.csv"
val_prices_loc = data_folder_loc + "val_prices.csv"

#join data
train['price_doc'] = prices['price_doc']

train_subset = train[train['id'] < 27235]
train_2014 = train_subset[train_subset['id'] >13572]
validation = train[train['id'] >= 27235]

val_prices = validation['price_doc']

validation = validation.drop('price_doc', 1)

train_2014.to_csv(train_subset_loc)
validation.to_csv(validation_loc)
val_prices.to_csv(val_prices_loc)