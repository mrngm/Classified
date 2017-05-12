# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:59:29 2017

@author: Thijs
"""

#%% Libraries and files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trainset = pd.read_csv("input/train.csv")

plt.figure()

#%% full_sq

plt.plot(trainset['full_sq'], trainset['price_doc'], "o",
         color="g", ms=5)
plt.show()

# Remove outlier
trainset2 = trainset[trainset['full_sq'] < 4000]

sns.jointplot(x="full_sq", y="price_doc", data=trainset2,
              color="g", size=8, s=10)

#%% life_sq

plt.plot(trainset['life_sq'], trainset['price_doc'], "o",
         color="g", ms=5)
plt.show()

# Remove outlier
trainset2 = trainset[trainset['life_sq'] < 7000]

sns.jointplot(x="life_sq", y="price_doc", data=trainset2,
              color="g", size=8, s=10)

#%% num_room

plt.figure()
sns.countplot(x="num_room", data=trainset)
plt.xticks(rotation='vertical')
plt.show()

#%% build_year

buildyears = np.sort(trainset.build_year.unique())

plt.figure()
plt.plot(trainset['build_year'], trainset['price_doc'], "o",
         color="g", ms=5)
plt.show()

# Remove outliers
trainset2 = trainset[(trainset.build_year > 1600)
    & (trainset.build_year != 4965)
    & (trainset.build_year != 20052009)]

sns.jointplot(x="build_year", y="price_doc", data=trainset2,
              color="g", size=8, s=10)