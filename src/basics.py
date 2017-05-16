import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATADIR = "../data/"

macrofile = DATADIR + "macro.csv"
trainfile = DATADIR + "train.csv"
testfile  = DATADIR + "test.csv"

macrodata = pd.read_csv(macrofile)
traindata = pd.read_csv(trainfile)
testdata  = pd.read_csv(testfile)

# make sure we show ALL the data
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_rows', None)


floor_max_floors = traindata.loc[:,['floor','max_floor']]

plt.figure()
floor_max_floors.query('floor <= max_floor') \
                .plot.scatter(x='floor', y='max_floor')
plt.show()