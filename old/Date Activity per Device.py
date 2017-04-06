# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 18:47:15 2017

@author: Gebruiker
"""

print 'importing libraries'
import pandas as pd
import cPickle as pickle
import numpy as np
import os
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from haversine import haversine

#%% Importing Data
print 'loading data'

#Assumes original kaggle files are in datadir
datadir = 'D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                     index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                        usecols=['event_id','app_id','is_active'],
                        dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

#%%Device Time
print 'Splitting timestamp within events and device time activity'

events = events.assign(
    date_year = lambda d: d['timestamp'].dt.year,
    date_month = lambda d: d['timestamp'].dt.month,
    date_week = lambda d: d['timestamp'].dt.week,
    date_day = lambda d: d['timestamp'].dt.day,
    date_hour = lambda d: d['timestamp'].dt.hour,
    date_minute = lambda d: d['timestamp'].dt.minute,
    date_second = lambda d: d['timestamp'].dt.second,
    date_weekday = lambda d: d['timestamp'].dt.weekday,
)

Day = events[['device_id', 'date_day', 'date_year']]
Day_Ac = Day.groupby(['device_id', 'date_day']).count()
Day_Ac = Day_Ac.unstack()
Day_Ac = Day_Ac.fillna(0)
Day_Ac = Day_Ac.div(Day_Ac.sum(axis=1), axis=0)

Week = events[['device_id', 'date_week', 'date_year']]
Week_Ac = Week.groupby(['device_id', 'date_week']).count()
Week_Ac = Week_Ac.unstack()
Week_Ac = Week_Ac.fillna(0)
Week_Ac = Week_Ac.div(Week_Ac.sum(axis=1), axis=0)

Hour = events[['device_id', 'date_hour', 'date_year']]
Total_Hour_Ac = Hour.groupby(['device_id', 'date_hour']).count()
Total_Hour_Ac = Total_Hour_Ac.unstack()
Total_Hour_Ac = Total_Hour_Ac.fillna(0)
Total_Hour_Ac = Total_Hour_Ac.div(Total_Hour_Ac.sum(axis=1), axis=0)

#%% Device Activity Per Hour Per Day
print 'Device Activity Per Hour Per Day, Normalized'

day = events[['device_id','date_hour','date_weekday']]
Mon = day.loc[day['date_weekday'] == 0]
Tue = day.loc[day['date_weekday'] == 1]
Wed = day.loc[day['date_weekday'] == 2]
Thu = day.loc[day['date_weekday'] == 3]
Fri = day.loc[day['date_weekday'] == 4]
Sat = day.loc[day['date_weekday'] == 5]
Sun = day.loc[day['date_weekday'] == 6]

Mon_Ac = Mon.groupby(['device_id', 'date_hour']).count()
Mon_Ac = Mon_Ac.unstack()
Mon_Ac = Mon_Ac.fillna(0)
Mon_Ac = Mon_Ac.div(Mon_Ac.sum(axis=1), axis=0)

Tue_Ac = Tue.groupby(['device_id', 'date_hour']).count()
Tue_Ac = Tue_Ac.unstack()
Tue_Ac = Tue_Ac.fillna(0)
Tue_Ac = Tue_Ac.div(Tue_Ac.sum(axis=1), axis=0)

Wed_Ac = Wed.groupby(['device_id', 'date_hour']).count()
Wed_Ac = Wed_Ac.unstack()
Wed_Ac = Wed_Ac.fillna(0)
Wed_Ac = Wed_Ac.div(Wed_Ac.sum(axis=1), axis=0)

Thu_Ac = Thu.groupby(['device_id', 'date_hour']).count()
Thu_Ac = Thu_Ac.unstack()
Thu_Ac = Thu_Ac.fillna(0)
Thu_Ac = Thu_Ac.div(Thu_Ac.sum(axis=1), axis=0)

Fri_Ac = Fri.groupby(['device_id', 'date_hour']).count()
Fri_Ac = Fri_Ac.unstack()
Fri_Ac = Fri_Ac.fillna(0)
Fri_Ac = Fri_Ac.div(Fri_Ac.sum(axis=1), axis=0)

Sat_Ac = Sat.groupby(['device_id', 'date_hour']).count()
Sat_Ac = Sat_Ac.unstack()
Sat_Ac = Sat_Ac.fillna(0)
Sat_Ac = Sat_Ac.div(Sat_Ac.sum(axis=1), axis=0)

Sun_Ac = Sun.groupby(['device_id', 'date_hour']).count()
Sun_Ac = Sun_Ac.unstack()
Sun_Ac = Sun_Ac.fillna(0)
Sun_Ac = Sun_Ac.div(Sun_Ac.sum(axis=1), axis=0)
#%% Concatenating all days
print 'concatenating all days per hour'

General_Week_Ac = pd.concat([Mon_Ac, Tue_Ac, Wed_Ac, Thu_Ac, Fri_Ac, Sat_Ac, Sun_Ac], axis=1)
General_Week_Ac = General_Week_Ac.fillna(0)
#%% Device Activity per Week
print 'Device Activity per Week'

week = events[['device_id','date_hour','date_week']]
W17 = week.loc[week['date_week'] == 17]
W18 = week.loc[week['date_week'] == 18]

W17_Ac = W17.groupby(['device_id', 'date_hour']).count()
W17_Ac = W17_Ac.unstack()
W17_Ac = W17_Ac.fillna(0)
W17_Ac = W17_Ac.div(W17_Ac.sum(axis=1), axis=0)

W18_Ac = W18.groupby(['device_id', 'date_hour']).count()
W18_Ac = W18_Ac.unstack()
W18_Ac = W18_Ac.fillna(0)
W18_Ac = W18_Ac.div(W18_Ac.sum(axis=1), axis=0)

#%% Concatenating all days
print 'concatenating all weeks per hour'

W1718_Ac = pd.concat([W17_Ac, W18_Ac], axis=1)
W1718_Ac = W1718_Ac.fillna(0)

#%% Turning Dataframe into csr matrix
print 'Turning all dataframes into csr matrices'

Day_AcM = csr_matrix(Day_Ac)
Week_AcM = csr_matrix(Week_Ac)
Total_Hour_AcM = csr_matrix(Total_Hour_Ac)
Mon_AcM = csr_matrix(Mon_Ac)
Tue_AcM = csr_matrix(Tue_Ac)
Wed_AcM = csr_matrix(Wed_Ac)
Thu_AcM = csr_matrix(Thu_Ac)
Fri_AcM = csr_matrix(Fri_Ac)
Sat_AcM = csr_matrix(Sat_Ac)
Sun_AcM = csr_matrix(Sun_Ac)
General_Week_AcM = csr_matrix(General_Week_Ac)
W17_AcM = csr_matrix(W17_Ac)
W18_AcM = csr_matrix(W18_Ac)
W1718_AcM = csr_matrix(W1718_Ac)

#%% Save data

#Save train data
traindir = './Train_features/'
pickle.dump(Day_AcM, open(traindir + 'Normalized_Day_Active.p',"wb"))
pickle.dump(Week_AcM, open(traindir + 'Normalized-Week_Active.p',"wb"))
pickle.dump(Total_Hour_AcM, open(traindir + 'Normalized_Hour_Active.p',"wb"))
pickle.dump(Mon_AcM, open(traindir + 'Monday_Device_Activity.p',"wb"))
pickle.dump(Tue_AcM, open(traindir + 'Tuesday_Device_Activity.p',"wb"))
pickle.dump(Wed_AcM, open(traindir + 'Wednesday_Device_Activity.p',"wb"))
pickle.dump(Thu_AcM, open(traindir + 'Thursday_Device_Activity.p',"wb"))
pickle.dump(Fri_AcM, open(traindir + 'Friday_Device_Activity.p',"wb"))
pickle.dump(Sat_AcM, open(traindir + 'Saturday_Device_Activity.p',"wb"))
pickle.dump(Sun_AcM, open(traindir + 'Sunday_Device_Activity.p',"wb"))
pickle.dump(General_Week_AcM, open(traindir + 'Week_Device_Activity.p',"wb"))
pickle.dump(W17_AcM, open(traindir + 'Week_17_Activity.p',"wb"))
pickle.dump(W18_AcM, open(traindir + 'Week_18_Activity.p',"wb"))
pickle.dump(W1718_AcM, open(traindir + 'Week_1718_Activity.p',"wb"))


