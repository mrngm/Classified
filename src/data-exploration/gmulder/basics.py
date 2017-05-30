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
leq = floor_max_floors.query('floor <= max_floor')
                #.plot.scatter(x='floor', y='max_floor', c='green')
gt = floor_max_floors.query('floor > max_floor')
                #.plot.scatter(x='floor', y='max_floor', c='red')
plt.scatter(y=leq['floor'], x=leq['max_floor'], c='green', marker='+', label='floor <= max_floor')
plt.scatter(y=gt['floor'], x=gt['max_floor'], c='red', marker='x', label='floor > max_floor')
plt.legend()
plt.ylabel('floor')
plt.xlabel('max_floor')
plt.show()
