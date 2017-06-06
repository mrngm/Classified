# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 16:06:30 2017

@author: Roel
"""

import pandas as pd

# Declare paths
data_folder_loc = "../data/"
train_loc = data_folder_loc + "train.csv"
train_subset_loc = data_folder_loc + "train_subset.csv"
validation_loc = data_folder_loc + "validation.csv"

train = pd.read_csv(train_loc)

#Split on instances before(<) and in(>=) 2015
train_subset = train[train['id'] < 27235]
validation = train[train['id'] >= 27235]

train_subset.to_csv(train_subset_loc)
validation.to_csv(validation_loc)