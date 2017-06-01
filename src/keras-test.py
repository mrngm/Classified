# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:38:02 2017

@author: Thijs
"""

#%% Libraries

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU

#%% Files

DATADIR = '../data/'
subdir = '../submission/'

trainfile   = DATADIR + 'one-hot_median_filled_train.csv'
trainprices = DATADIR + 'train_prices.csv'
testfile    = DATADIR + 'one-hot_median_filled_test.csv'

trainset    = pd.read_csv(trainfile)
priceset    = pd.read_csv(trainprices)
testset     = pd.read_csv(testfile)

X_train     = np.array(trainset)[:,1:]
Y_train     = np.array(priceset)[:,1:]
X_test      = np.array(testset)[:,1:]

#%% Model

input_shape = X_train.shape[1:]
model = Sequential()
model.add(Dense(20, kernel_initializer='normal', input_shape=input_shape))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

#%% Training

fit = model.fit(X_train, Y_train, epochs=15)

pred = model.predict(X_test)

#%% Save file
sub = pd.DataFrame(pred, columns=['price_doc'])
sub.index.names = ['id']

sub_file = 'DL_sub.csv'
sub.to_csv(subdir + sub_file, index=True)
print ('submission saved to ' + subdir + sub_file)