# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 14:44:49 2017

@author: Gebruiker
"""

import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import calendar
import datetime as dt
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from math import sin, cos, radians, atan2, sqrt

events = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/events.csv', parse_dates=['timestamp'])
app_events = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/app_events.csv',  usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
applabels = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/app_labels.csv')
labelcats = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/label_categories.csv', index_col='label_id',squeeze=True)
Universal = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Noise Eliminated Universal Files/Universal.csv', index_col = 0, parse_dates=['timestamp'], infer_datetime_format=True ,  dtype={'gender': object, 'group': object, 'phone_brand': object, 'device_model': object, 'category': str})
Date = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Noise Eliminated Universal Files/event_splitted_dt.csv', index_col = 0)
Groep = pd.read_excel('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/label_groep.xlsx')
app_event = pd.merge(left=Universal,right=Groep, how='left', left_on='label_id', right_on='label_id')

app_event = app_event.drop_duplicates(['device_id'], keep='first')

app_event_2 = app_event.copy(deep=True)
app_event = app_event[app_event.gender == 'F']
app_event = app_event[app_event.age > 35]
app_event.head(10)

app_event.shape
app_event.columns
app_event.timestamp=pd.to_datetime(app_event.timestamp)
app_event['time_hour'] = app_event.timestamp.apply(lambda x: x.hour)
app_event['time_hour'].value_counts()

#event frequency by XXXX
ax = sns.countplot(x="time_hour", data=app_event)
plt.title('Female Older Than 35 - Event Count Hour', fontsize=20)
app_event['week_day'] = app_event.timestamp.apply(lambda x: calendar.day_name[x.weekday()])
ax = sns.countplot(x="week_day", data=app_event)
plt.title('Female Older Than 35 - Event Count Weekday', fontsize=20)
ax = sns.countplot(x="phone_brand", data=app_event)
plt.title('Female Older Than 35 - Phone Brand', fontsize=20)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
ax = sns.countplot(x="Group", data=app_event)
plt.title('Female Older Than 35 - App Category', fontsize=20)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
sns.distplot(app_event.age.dropna(), color='#63EA55')
width = 1
axes = plt.gca()
axes.set_ylim([0.00,0.10])
axes.set_xlim([15,90])
plt.xlabel('Age')
plt.title('Female Age Distribution - Without Duplicates')
sns.despine()
plt.show
app_event = app_event_2.copy(deep=True)

plt.figure(figsize=(13,8))
plt.hist2d(app_event['device_id'].values, app_event['age'].values, bins=80)
plt.xlabel('Device ID', fontsize=15)
plt.ylabel('Age of user', fontsize=15)
plt.title('Distribution of ages based on device ID - 80 bins', fontsize=20)
plt.show()

gender_bin = (app_event['gender'] == 'M').tolist()

plt.figure(figsize=(13,5))
plt.hist2d(app_event['device_id'].values, gender_bin, bins=(50, 2))
plt.xlabel('Device ID', fontsize=15)
plt.ylabel('Female                   Male', fontsize=15)
plt.title('Distribution of gender based on device ID - 100 bins', fontsize=20)
plt.show()


List = app_event.Group.unique()
app_event_2 = app_event.copy(deep=True)
app_event = app_event[app_event.Group == 'Travel']


ax = sns.countplot(x="group", data=app_event)
plt.title('Universal Total Information - Travel - Age Group Distribution', fontsize=20)
app_event = app_event_2.copy(deep=True)

#Splitting Time Data
app_event = app_event.assign(
    date_year = lambda d: d['timestamp'].dt.year,
    date_month = lambda d: d['timestamp'].dt.month,
    date_day = lambda d: d['timestamp'].dt.day,
    date_hour = lambda d: d['timestamp'].dt.hour,
    date_minute = lambda d: d['timestamp'].dt.minute,
    date_second = lambda d: d['timestamp'].dt.second,
    date_weekday = lambda d: d['timestamp'].dt.weekday
)

app_event = app_event.drop_duplicates(['device_id'], keep='first')
app_event_2 = app_event.copy(deep=True)
app_event = app_event_2.copy(deep=True)
app_event = app_event[app_event.date_month == 5]


ax = sns.countplot(x='date_day', data=app_event)
plt.title('Universal Total Information - May - Day Distribution - Without Duplicates', fontsize=20)
ax.set_xticklabels(ax.xaxis.get_majorticklabels())

