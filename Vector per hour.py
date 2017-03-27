# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:05:24 2017

@author: Gebruiker
"""

import pandas as pd
import datetime
import cPickle as pickle

#Importing Dataset
Universal = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Noise Eliminated Universal Files/Universal.csv', index_col = 0, parse_dates=['timestamp'], infer_datetime_format=True ,  dtype={'gender': object, 'group': object, 'phone_brand': object, 'device_model': object, 'category': str})
Date = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Noise Eliminated Universal Files/event_splitted_dt.csv', index_col = 0)
Groep = pd.read_excel('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/label_groep.xlsx')
app_event = pd.merge(left=Universal,right=Groep, how='left', left_on='label_id', right_on='label_id')

app_event = app_event.assign(
    date_year = lambda d: d['timestamp'].dt.year,
    date_month = lambda d: d['timestamp'].dt.month,
    date_day = lambda d: d['timestamp'].dt.day,
    date_hour = lambda d: d['timestamp'].dt.hour,
    date_minute = lambda d: d['timestamp'].dt.minute,
    date_second = lambda d: d['timestamp'].dt.second,
    date_weekday = lambda d: d['timestamp'].dt.weekday
)

app_event.columns

s = app_event.groupby(['device_id', 'date_hour']).size()
m = s.unstack()


m = m.fillna(0)
print m
m_norm = m.div(m.sum(axis=1), axis=0)
k = m_norm.as_matrix()
d = m_norm.reset_index().values
Device_Normalized_Hour_DataFrame = m_norm
Device_Normalized_Hour_Array_With_Index = d
Device_Normalized_Hour_Array_Without_Index = k

with open(r"Device_Normalized_Hour_DataFrame.pickle", "wb") as output_file:
    pickle.dump(Device_Normalized_Hour_DataFrame, output_file)

with open(r"Device_Normalized_Hour_Array_With_Index.pickle", "wb") as output_file:
    pickle.dump(Device_Normalized_Hour_Array_With_Index, output_file)

with open(r"Device_Normalized_Hour_Array_Without_Index.pickle", "wb") as output_file:
    pickle.dump(Device_Normalized_Hour_Array_Without_Index, output_file)

