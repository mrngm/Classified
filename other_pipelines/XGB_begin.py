# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:24:00 2017
@author: Roel & Jeffrey 
"""
#%%
#Load libraries
import cPickle as pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp

# XGBoost params 
# https://www.kaggle.com/anokas/talkingdata-mobile-user-demographics/sparse-xgboost-starter-2-26857
# Might need considerable tweaking. These were taken from a Sparse XGBoost kernel implementation by one of the top contributors on Kaggle
params = {}
params['booster'] = 'gblinear'
params['objective'] = "multi:softprob"
params['eval_metric'] = 'mlogloss'
params['eta'] = 0.005
params['num_class'] = 12
params['lambda'] = 3
params['alpha'] = 2

apps_clean = pickle.load(open("stringdumpfinal.p","rb"))
#tfidf_features = pickle.load(open('termmat.p', 'rb'))
labels = pickle.load(open('labels.p', 'rb'))

#transform to numerical, one-hot, labels
lb = preprocessing.LabelBinarizer()
templabels=lb.fit_transform(labels) #also use to transform back
templabels = templabels*np.arange(templabels.shape[1])
numlabels = np.sum(templabels,axis=1)


# ------------ XGBoost -------------------------
# based on two kaggle kernels: https://www.kaggle.com/zfturbo/talkingdata-mobile-user-demographics/xgboost-simple-starter by ZFTurbo and the link above, by Anokas

# Random 10% for validation 
kf = list(StratifiedKFold(y, n_folds=10, shuffle=True, random_state=1337))[0]

Xtr, Xte = apps_clean[kf[0], :], apps_clean[kf[1], :]
ytr, yte = numlabels[kf[0]], numlabels[kf[1]]

print('Training set: ' + str(Xtr.shape))
print('Validation set: ' + str(Xte.shape))

d_train = xgb.DMatrix(Xtr, label=ytr)
d_valid = xgb.DMatrix(Xte, label=yte)

watchlist = [(d_train, 'train'), (d_valid, 'eval')]

clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=25)

pred = clf.predict(xgb.DMatrix(Xtest))

pred = pd.DataFrame(pred, columns=labels) #might need the index of the sample submission?
pred.head()
pred.to_csv('output_xgb.csv', index=True)
