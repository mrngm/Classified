# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:46:25 2017

@author: Thijs
"""
import csv
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

file = '../MLiP/testset.csv'                # test set with ~300.000 records
file = '../MLiP/event_splitted_dt.csv'

with open(file, 'r') as f:
  dr = csv.DictReader(f, delimiter=',')
  dataset = list(dr)
  
times = []

for i in dataset:
    times.append(datetime.datetime.strptime(i['date_day'] + "-" + i['date_month'] + "-" + i['date_year'] + " " + i['date_hour'] + ":" + i['date_minute'] + ":" + i['date_second'], "%d-%m-%Y %H:%M:%S").date())
    
print(len(times))

mpl_data = mdates.date2num(times)

fig, ax = plt.subplots(1,1)
ax.hist(mpl_data)
fig.autofmt_xdate()
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
#ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
plt.show()