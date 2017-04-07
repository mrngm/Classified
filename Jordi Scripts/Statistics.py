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
from haversine import haversine


gatrain = pd.read_csv('D:\School\School\Master\Jaar_1\Machine Learning in Practice\Competition\Data\Mobile Data\gender_age_train.csv', index_col='device_id')
gatest = pd.read_csv('D:\School\School\Master\Jaar_1\Machine Learning in Practice\Competition\Data\Mobile Data\gender_age_test.csv', index_col = 'device_id')
phone = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/phone_brand_device_model.csv')
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
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

Phone_brand = app_event.groupby('phone_brand').count()
Event = app_event.groupby('event_id').count()
Device = app_event.groupby('device_model').count()
Category = app_event.groupby('category').count()
Gender = app_event.groupby('gender').count()
Age = app_event.groupby('age').count()
Device_Info = app_event.groupby(['phone_brand','device_model']).count()
Location = app_event.groupby(['device_id','longitude', 'latitude']).count()
App_Activity = app_event.groupby(['category', 'is_installed','is_active']).count()
App_Activity_Gender = app_event.groupby(['gender', 'Group']).count()

Phone_brand['index1'] = Phone_brand.index
Device_Info['index1'] = Device_Info.index
Location['index1'] = Location.index
App_Activity_Gender['index1'] = App_Activity_Gender.index

ListP = Phone_brand.index1.unique()
ListD = Location.index1.unique()
ListL = Device_Info.index1.unique()
ListG = app_event.Group.unique()
ListAAG = App_Activity_Gender.index1.unique()

#Distance Calculator
CSV_PATH_PREFIX = 'D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/MLiP/'
CSV_SRC_PREFIX = 'D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/unzipped/'

events = app_event

distance_events = pd.DataFrame(columns=["device_id", "date", "day_distance"])

events.sort_values(['device_id', 'timestamp'], ascending=[True, True], inplace=True)

per_device = events.groupby('device_id')

for device, device_events in per_device:
    per_day = device_events.groupby(device_events['timestamp'].dt.date)
    #print "{}".format(device)
    num_days = 0
    days_distance = []
    for date, date_events in per_day:
        prev_lat = None
        prev_long = None
        total_distance = 0.0
        for ev in date_events.itertuples():
            if ev.latitude == 0.0 or ev.longitude == 0.0:
                continue
            if prev_lat == None and prev_long == None:
                prev_lat  = float(ev.latitude)
                prev_long = float(ev.longitude)
                continue
            if prev_lat == ev.latitude and prev_long == ev.longitude:
                continue
            total_distance += haversine((prev_lat, prev_long), (float(ev.latitude), float(ev.longitude)))
            prev_lat  = float(ev.latitude)
            prev_long = float(ev.longitude)
        #print "  {}: {}".format(date, total_distance)
        days_distance.append(total_distance)
        distance_events = distance_events.append({"device_id": str(device), "date": date, "day_distance": total_distance}, ignore_index=True)

    #mean_dist = round(np.mean(days_distance), 2)
    #std_dist  = round(np.std(days_distance), 2)

if not os.path.isfile(CSV_PATH_PREFIX + 'device_distance.csv'):
    events.to_csv(CSV_PATH_PREFIX + 'device_distance.csv', index=False)

if not os.path.isfile(CSV_PATH_PREFIX + 'device_day_distance.csv'):
    distance_events.to_csv(CSV_PATH_PREFIX + 'device_day_distance.csv', index=False)