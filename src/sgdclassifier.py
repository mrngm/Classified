#!/usr/bin/env python2

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np


DATADIR = "../data/"

# check for cleaned data files
try:
    trainfn = "one-hot_median_filled_train.csv"
    pricefn = "train_prices.csv"
    testfn  = "one-hot_median_filled_test.csv"
    trainfile = pd.read_csv(DATADIR + trainfn)
    pricefile = pd.read_csv(DATADIR + pricefn)
    testfile  = pd.read_csv(DATADIR + testfn)
except IOError as e: 
    print "Could not read file: {}".format(e)

sc = StandardScaler()

clf = SGDClassifier(loss="hinge", penalty="l2")
tf = trainfile[["life_sq", 'build_year', 'num_room', 'raion_popul', 'shopping_centers_raion']]
sc.fit(tf)
tf = sc.transform(tf)

tt = sc.transform(testfile[["life_sq", 'build_year', 'num_room', 'raion_popul', 'shopping_centers_raion']])

clf.fit(tf, pricefile['price_doc'])

pred = clf.predict(tt)
testids = np.arange(30474, 38136)

print pred
print testids

fin = pd.DataFrame({"id": testids, "price_doc": pred})

print fin

fin.to_csv("../data/sgdregression.csv", index=False, index_label=False, header=True)

# vim: set et:ts=4:sw=4:
