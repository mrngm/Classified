# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:05:24 2017

@author: Gebruiker
"""

import pandas as pd
import datetime
import cPickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

#Importing Dataset
Universal = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Noise Eliminated Universal Files/Universal.csv', index_col = 0, parse_dates=['timestamp'], infer_datetime_format=True)
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
Universal.columns
app_event.columns

DD = app_event.groupby(['device_id', 'date_hour']).size()
DD = DD.unstack()

GDH = app_event.groupby(['group', 'date_hour']).size()
GDH = GDH.unstack()

CDH = app_event.groupby(['category', 'date_hour']).size()
CDH = CDH.unstack()

GC = app_event.groupby(['group', 'category']).size()
GC = GC.unstack()

GG = app_event.groupby(['group', 'Group']).size()
GG = GG.unstack()

AC = app_event.groupby(['app_id', 'category']).size()
AC = AC.unstack()

GD = app_event.groupby(['group', 'device_model']).size()
GD = GD.unstack()

AP = app_event.groupby(['age', 'phone_brand']).size()
AP = AP.unstack()

GRPB = app_event.groupby(['gender', 'phone_brand']).size()
GRPB = GRPB.unstack()

GRDM = app_event.groupby(['gender', 'device_model']).size()
GRDM = GRDM.unstack()

GPB = app_event.groupby(['group', 'phone_brand']).size()
GPB = GPB.unstack()

ADH = app_event.groupby(['age', 'date_hour']).size()
ADH = ADH.unstack()

GRDH = app_event.groupby(['gender', 'date_hour']).size()
GRDH = GRDH.unstack()

Glong = app_event.groupby(['group', 'longitude']).size()
Glong = Glong.unstack()

Glat = app_event.groupby(['group', 'latitude']).size()
Glat = Glat.unstack()

app_event_2 = app_event.copy(deep=True)
app_event = app_event.drop_duplicates(['device_id'], keep='first')
app_event = app_event_2.copy(deep=True)


DD = DD.fillna(0)
GDH = GDH.fillna(0)
CDH = CDH.fillna(0)
GC = GC.fillna(0)
GG = GG.fillna(0)
AC = AC.fillna(0)
GD = GD.fillna(0)
AP = AP.fillna(0)
ADH = ADH.fillna(0)
GRDH = GRDH.fillna(0)
GRPB = GRPB.fillna(0)
GPB = GPB.fillna(0)
GRDM = GRDM.fillna(0)
Glong = Glong.fillna(0)
Glat = Glat.fillna(0)

DD_norm = DD.div(DD.sum(axis=1), axis=0)
GDH_norm = GDH.div(GDH.sum(axis=1), axis=0)
CDH_norm = CDH.div(CDH.sum(axis=1), axis=0)
GC_norm = GC.div(GC.sum(axis=1), axis=0)
GG_norm = GG.div(GG.sum(axis=1), axis=0)
AC_norm = AC.div(AC.sum(axis=1), axis=0)
GD_norm = GD.div(GD.sum(axis=1), axis=0)
AP_norm = AP.div(AP.sum(axis=1), axis=0)
ADH_norm = ADH.div(ADH.sum(axis=1), axis=0)
GRDH_norm = GRDH.div(GRDH.sum(axis=1), axis=0)
GRPB_norm = GRPB.div(GRPB.sum(axis=1), axis=0)
GPB_norm = GPB.div(GPB.sum(axis=1), axis=0)
GRDM_norm = GRDM.div(GRDM.sum(axis=1), axis=0)
Glong_norm = Glong.div(Glong.sum(axis=1), axis=0)
Glat_norm = Glat.div(Glat.sum(axis=1), axis=0)

DD_stand = (DD - DD.mean()) / (DD.max() - DD.min())
GDH_stand = (GDH - GDH.mean()) / (GDH.max() - GDH.min())
CDH_stand = (CDH - CDH.mean()) / (CDH.max() - CDH.min())
GC_stand = (GC - GC.mean()) / (GC.max() - GC.min())
GG_stand = (GG - GG.mean()) / (GG.max() - GG.min())
AC_stand = (AC - AC.mean()) / (AC.max() - AC.min())
GD_stand = (GD - GD.mean()) / (GD.max() - GD.min())
AP_stand = (AP - AP.mean()) / (AP.max() - AP.min())
ADH_stand = (ADH - ADH.mean()) / (ADH.max() - ADH.min())
GRDH_stand = (GRDH - GRDH.mean()) / (GRDH.max() - GRDH.min())
GRPB_stand = (GRPB - GRPB.mean()) / (GRPB.max() - GRPB.min())
GPB_stand = (GPB - GPB.mean()) / (GPB.max() - GPB.min())
GRDM_stand = (GRDM - GRDM.mean()) / (GRDM.max() - GRDM.min())
Glong_stand = (Glong - Glong.mean()) / (Glong.max() - Glong.min())
Glat_stand = (Glat - Glat.mean()) / (Glat.max() - Glat.min())
"""
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
"""





    

