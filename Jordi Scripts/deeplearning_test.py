# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:27:35 2017

@author: Thijs
"""
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense

file = "../MLiP/dl_testset.csv"
#file = "../MLiP/events_combined.csv"

with open(file, 'r', encoding="utf8") as f:
  dr = csv.DictReader(f, delimiter=',')
  dataset = list(dr)
  
X = []
Y = []
for line in dataset:
    x = [line['app_label_id'], line['device_age']
    #, line['event_latitude'], line['event_longitude']
    ]
    y = line['device_gender']
    if not ('XXX' in x or y == 'XXX'):
        X.append(x)
        Y.append(y)
    
X = np.array(X)
Y = np.array(Y)
Y[Y == 'M'] = 1
Y[Y == 'F'] = 0
 
model = Sequential()
model.add(Dense(4, input_dim=len(X[0]), init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, nb_epoch=10, batch_size=10)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)

rounded = [round(x[0]) for x in predictions]
print(rounded)