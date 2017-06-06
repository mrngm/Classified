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
from sklearn.model_selection import StratifiedKFold as SKF

#%% Code for fiting from generator by Chenglong Chen
# (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)

def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
            
def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :]
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

#%% Files

DATADIR = '../data/'
subdir = '../submission/'

trainfile   = DATADIR + 'one-hot_median_filled_train.csv'
trainprices = DATADIR + 'train_prices.csv'
testfile    = DATADIR + 'one-hot_median_filled_test.csv'

trainset    = pd.read_csv(trainfile)
priceset    = pd.read_csv(trainprices)
testset     = pd.read_csv(testfile)

X_train_full    = np.array(trainset)[:,1:]
labels          = np.array(priceset)[:,1:]
X_test_full     = np.array(testset)[:,1:]
indexes         = testset.iloc[:,0]

#%% Model

def base_model():
    input_shape = X_train_full.shape[1:]
    model = Sequential()
    model.add(Dense(100, kernel_initializer='normal', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(75, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#%%
#Create cross-validation splits (n-fold)
n = 5
skf1 = SKF(n_splits = n)

#%%
#Cross validating loop
test_acc = np.zeros(n)
loss = np.zeros(n)

#%% Training

model = base_model()
i = 0

for train, test in skf1.split(X_train_full, labels[:,0]):
    print ('start iteration ' + str(i+1))
    
    X_train = X_train_full[train]
    X_test  = X_train_full[test]
    
    Y_train = labels[train]
    Y_test  = labels[test]
    
    model.fit_generator(generator=batch_generator(X_train, Y_train, 100, True),
                        epochs=5,
                        validation_data=(X_test, Y_test),
                        steps_per_epoch=2000)
    #model.fit(X_train[train], labels[train],
    #          validation_data=(X_train[test],labels[test]),
    #          epochs=5, batch_size=10)
    i+=1

pred = model.predict_generator(generator=batch_generatorp(X_test_full, 200, False),
                               steps=X_test_full.shape[0])

#%% Save file
sub = pd.DataFrame(pred, index=indexes, columns=['price_doc'])
sub.index.names = ['id']

sub_file = 'DL_sub.csv'
sub.to_csv(subdir + sub_file, index=True)
print ('submission saved to ' + subdir + sub_file)