print 'importing libraries'
import pandas as pd
import cPickle as pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from haversine import haversine

#%% Forked from dune_dweller on Kaggle: https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/a-linear-model-on-apps-and-labels
print 'loading data'

#Assumes original kaggle files are in datadir
datadir = './input'
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

#%%
print 'extracting features'

#One-hot encoded phone brands
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))

#One-hot encoded phone models
m = phone.phone_brand.str.cat(phone.device_model)
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))

#Boolean encoding of installed apps (might be rows consisting of just 0 values because of missing events)
appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
deviceapps.head()

d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))

#Boolean encoding of installed apps (might be rows consisting of just 0 values because of missing events)
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)

devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelabels.head()

d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))
#%% Device Event Location
print 'extracting features Device per longitude and latitude'

events = events[events.longitude != 0]
events = events[events.latitude != 0]
distance = events.copy(deep=True)
longcoder = LabelEncoder().fit(events.longitude)
latcoder = LabelEncoder().fit(events.latitude)

events['longitude'] = longcoder.transform(events['longitude'])
events['latitude'] = latcoder.transform(events['latitude'])
nlongitude = len(longcoder.classes_)
nlatitude = len(latcoder.classes_)

devicelong = (events[['device_id','longitude']]
                .groupby(['device_id','longitude'])['longitude'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelong.head()

d = devicelong.dropna(subset=['trainrow'])
Xtr_long = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.longitude)), 
                      shape=(gatrain.shape[0], nlongitude))
d = devicelong.dropna(subset=['testrow'])
Xte_long = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.longitude)), 
                      shape=(gatest.shape[0], nlongitude))

devicelat = (events[['device_id','latitude']]
                .groupby(['device_id','latitude'])['latitude'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelat.head()

d = devicelat.dropna(subset=['trainrow'])
Xtr_lat = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.latitude)), 
                      shape=(gatrain.shape[0], nlatitude))
d = devicelat.dropna(subset=['testrow'])
Xte_lat = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.latitude)), 
                      shape=(gatest.shape[0], nlatitude))

print('Latitude data: train shape {}, test shape {}'.format(Xtr_lat.shape, Xte_lat.shape))
print('Longitude data: train shape {}, test shape {}'.format(Xtr_long.shape, Xte_long.shape))
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

daycoder = LabelEncoder().fit(events.date_day)
hourcoder = LabelEncoder().fit(events.date_hour)
weekdaycoder = LabelEncoder().fit(events.date_weekday)

events['date_day'] = daycoder.transform(events['date_day'])
events['date_hour'] = hourcoder.transform(events['date_hour'])
events['date_weekday'] = weekdaycoder.transform(events['date_weekday'])
nday = len(daycoder.classes_)
nhour = len(hourcoder.classes_)
nweekday = len(weekdaycoder.classes_)

deviceday = (events[['device_id','date_day']]
                .groupby(['device_id','date_day'])['date_day'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
deviceday.head()

d = deviceday.dropna(subset=['trainrow'])
Xtr_day = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.date_day)), 
                      shape=(gatrain.shape[0], nday))
d = deviceday.dropna(subset=['testrow'])
Xte_day = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.date_day)), 
                      shape=(gatest.shape[0], nday))

devicehour = (events[['device_id','date_hour']]
                .groupby(['device_id','date_hour'])['date_hour'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicehour.head()

d = devicehour.dropna(subset=['trainrow'])
Xtr_hour = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.date_hour)), 
                      shape=(gatrain.shape[0], nhour))
d = devicehour.dropna(subset=['testrow'])
Xte_hour = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.date_hour)), 
                      shape=(gatest.shape[0], nhour))

deviceweekday = (events[['device_id','date_weekday']]
                .groupby(['device_id','date_weekday'])['date_weekday'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
deviceweekday.head()

d = deviceweekday.dropna(subset=['trainrow'])
Xtr_weekday = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.date_weekday)), 
                      shape=(gatrain.shape[0], nweekday))
d = deviceweekday.dropna(subset=['testrow'])
Xte_weekday = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.date_weekday)), 
                      shape=(gatest.shape[0], nweekday))

print('Day data: train shape {}, test shape {}'.format(Xtr_day.shape, Xte_day.shape))
print('Hour data: train shape {}, test shape {}'.format(Xtr_hour.shape, Xte_hour.shape))
print('Weekday data: train shape {}, test shape {}'.format(Xtr_weekday.shape, Xte_weekday.shape))
#%% Device Activity Per Hour Per Day
print 'Device Activity Per Hour Per Day, Normalized'

day = events[['device_id','date_hour','date_day']]
Mon = day.loc[day['date_day'] == 0]
Tue = day.loc[day['date_day'] == 1]
Wed = day.loc[day['date_day'] == 2]
Thu = day.loc[day['date_day'] == 3]
Fri = day.loc[day['date_day'] == 4]
Sat = day.loc[day['date_day'] == 5]
Sun = day.loc[day['date_day'] == 6]

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
#%% Creating Pickle objects per day
print 'Creating Pickle Objects per hour activity per day per device'

Mon_Ac = (Mon_Ac
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
Mon_Ac.head()
Mon_Ac.columns = ['device_id', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                  'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                  'h19', 'h20','h21', 'h22', 'h23', 'trainrow', 'testrow']
Mon_Ac.columns

Mon_Ac_Tr = Mon_Ac.copy(deep=True)
Mon_Ac_Tr.drop('device_id', inplace=True, axis=1)
Mon_Ac_Tr.drop('testrow', inplace=True, axis=1)
Mon_Ac_Tr = Mon_Ac_Tr.fillna(0)
Mon_Ac_Tr = Mon_Ac_Tr[Mon_Ac_Tr.trainrow != 0]
cols = Mon_Ac_Tr.columns.tolist()
cols = cols[-1:] + cols[:-1]
Mon_Ac_Tr = Mon_Ac_Tr[cols]
Xtr_Mon = csr_matrix(Mon_Ac_Tr.values)

Mon_Ac_Te = Mon_Ac.copy(deep=True)
Mon_Ac_Te.drop('device_id', inplace=True, axis=1)
Mon_Ac_Te.drop('trainrow', inplace=True, axis=1)
Mon_Ac_Te = Mon_Ac_Te.fillna(0)
Mon_Ac_Te = Mon_Ac_Te[Mon_Ac_Te.testrow != 0]
cols = Mon_Ac_Te.columns.tolist()
cols = cols[-1:] + cols[:-1]
Mon_Ac_Te = Mon_Ac_Te[cols]
Xte_Mon = csr_matrix(Mon_Ac_Te.values)

Tue_Ac = (Tue_Ac
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
Tue_Ac.head()
Tue_Ac.columns = ['device_id', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                  'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                  'h19', 'h20','h21', 'h22', 'h23', 'trainrow', 'testrow']
Tue_Ac.columns

Tue_Ac_Tr = Tue_Ac.copy(deep=True)
Tue_Ac_Tr.drop('device_id', inplace=True, axis=1)
Tue_Ac_Tr.drop('testrow', inplace=True, axis=1)
Tue_Ac_Tr = Tue_Ac_Tr.fillna(0)
Tue_Ac_Tr = Tue_Ac_Tr[Tue_Ac_Tr.trainrow != 0]
cols = Tue_Ac_Tr.columns.tolist()
cols = cols[-1:] + cols[:-1]
Tue_Ac_Tr = Tue_Ac_Tr[cols]
Xtr_Tue = csr_matrix(Tue_Ac_Tr.values)

Tue_Ac_Te = Tue_Ac.copy(deep=True)
Tue_Ac_Te.drop('device_id', inplace=True, axis=1)
Tue_Ac_Te.drop('trainrow', inplace=True, axis=1)
Tue_Ac_Te = Tue_Ac_Te.fillna(0)
Tue_Ac_Te = Tue_Ac_Te[Tue_Ac_Te.testrow != 0]
cols = Tue_Ac_Te.columns.tolist()
cols = cols[-1:] + cols[:-1]
Tue_Ac_Te = Tue_Ac_Te[cols]
Xte_Tue = csr_matrix(Tue_Ac_Te.values)

Wed_Ac = (Wed_Ac
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
Wed_Ac.head()
Wed_Ac.columns = ['device_id', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                  'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                  'h19', 'h20','h21', 'h22', 'h23', 'trainrow', 'testrow']
Wed_Ac.columns

Wed_Ac_Tr = Wed_Ac.copy(deep=True)
Wed_Ac_Tr.drop('device_id', inplace=True, axis=1)
Wed_Ac_Tr.drop('testrow', inplace=True, axis=1)
Wed_Ac_Tr = Wed_Ac_Tr.fillna(0)
Wed_Ac_Tr = Wed_Ac_Tr[Wed_Ac_Tr.trainrow != 0]
cols = Wed_Ac_Tr.columns.tolist()
cols = cols[-1:] + cols[:-1]
Wed_Ac_Tr = Wed_Ac_Tr[cols]
Xtr_Wed = csr_matrix(Wed_Ac_Tr.values)

Wed_Ac_Te = Wed_Ac.copy(deep=True)
Wed_Ac_Te.drop('device_id', inplace=True, axis=1)
Wed_Ac_Te.drop('trainrow', inplace=True, axis=1)
Wed_Ac_Te = Wed_Ac_Te.fillna(0)
Wed_Ac_Te = Wed_Ac_Te[Wed_Ac_Te.testrow != 0]
cols = Wed_Ac_Te.columns.tolist()
cols = cols[-1:] + cols[:-1]
Wed_Ac_Te = Wed_Ac_Te[cols]
Xte_Wed = csr_matrix(Wed_Ac_Te.values)

Thu_Ac = (Thu_Ac
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
Thu_Ac.head()
Thu_Ac.columns = ['device_id', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                  'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                  'h19', 'h20','h21', 'h22', 'h23', 'trainrow', 'testrow']
Thu_Ac.columns

Thu_Ac_Tr = Thu_Ac.copy(deep=True)
Thu_Ac_Tr.drop('device_id', inplace=True, axis=1)
Thu_Ac_Tr.drop('testrow', inplace=True, axis=1)
Thu_Ac_Tr = Thu_Ac_Tr.fillna(0)
Thu_Ac_Tr = Thu_Ac_Tr[Thu_Ac_Tr.trainrow != 0]
cols = Thu_Ac_Tr.columns.tolist()
cols = cols[-1:] + cols[:-1]
Thu_Ac_Tr = Thu_Ac_Tr[cols]
Xtr_Thu = csr_matrix(Thu_Ac_Tr.values)

Thu_Ac_Te = Thu_Ac.copy(deep=True)
Thu_Ac_Te.drop('device_id', inplace=True, axis=1)
Thu_Ac_Te.drop('trainrow', inplace=True, axis=1)
Thu_Ac_Te = Thu_Ac_Te.fillna(0)
Thu_Ac_Te = Thu_Ac_Te[Thu_Ac_Te.testrow != 0]
cols = Thu_Ac_Te.columns.tolist()
cols = cols[-1:] + cols[:-1]
Thu_Ac_Te = Thu_Ac_Te[cols]
Xte_Thu = csr_matrix(Thu_Ac_Te.values)

Fri_Ac = (Fri_Ac
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
Fri_Ac.head()
Fri_Ac.columns = ['device_id', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                  'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                  'h19', 'h20','h21', 'h22', 'h23', 'trainrow', 'testrow']
Fri_Ac.columns

Fri_Ac_Tr = Fri_Ac.copy(deep=True)
Fri_Ac_Tr.drop('device_id', inplace=True, axis=1)
Fri_Ac_Tr.drop('testrow', inplace=True, axis=1)
Fri_Ac_Tr = Fri_Ac_Tr.fillna(0)
Fri_Ac_Tr = Fri_Ac_Tr[Fri_Ac_Tr.trainrow != 0]
cols = Fri_Ac_Tr.columns.tolist()
cols = cols[-1:] + cols[:-1]
Fri_Ac_Tr = Fri_Ac_Tr[cols]
Xtr_Fri = csr_matrix(Fri_Ac_Tr.values)

Fri_Ac_Te = Fri_Ac.copy(deep=True)
Fri_Ac_Te.drop('device_id', inplace=True, axis=1)
Fri_Ac_Te.drop('trainrow', inplace=True, axis=1)
Fri_Ac_Te = Fri_Ac_Te.fillna(0)
Fri_Ac_Te = Fri_Ac_Te[Fri_Ac_Te.testrow != 0]
cols = Fri_Ac_Te.columns.tolist()
cols = cols[-1:] + cols[:-1]
Fri_Ac_Te = Fri_Ac_Te[cols]
Xte_Fri = csr_matrix(Fri_Ac_Te.values)

Sat_Ac = (Sat_Ac
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
Sat_Ac.head()
Sat_Ac.columns = ['device_id', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                  'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                  'h19', 'h20','h21', 'h22', 'h23', 'trainrow', 'testrow']
Sat_Ac.columns

Sat_Ac_Tr = Sat_Ac.copy(deep=True)
Sat_Ac_Tr.drop('device_id', inplace=True, axis=1)
Sat_Ac_Tr.drop('testrow', inplace=True, axis=1)
Sat_Ac_Tr = Sat_Ac_Tr.fillna(0)
Sat_Ac_Tr = Sat_Ac_Tr[Sat_Ac_Tr.trainrow != 0]
cols = Sat_Ac_Tr.columns.tolist()
cols = cols[-1:] + cols[:-1]
Sat_Ac_Tr = Sat_Ac_Tr[cols]
Xtr_Sat = csr_matrix(Sat_Ac_Tr.values)

Sat_Ac_Te = Sat_Ac.copy(deep=True)
Sat_Ac_Te.drop('device_id', inplace=True, axis=1)
Sat_Ac_Te.drop('trainrow', inplace=True, axis=1)
Sat_Ac_Te = Sat_Ac_Te.fillna(0)
Sat_Ac_Te = Sat_Ac_Te[Sat_Ac_Te.testrow != 0]
cols = Sat_Ac_Te.columns.tolist()
cols = cols[-1:] + cols[:-1]
Sat_Ac_Te = Sat_Ac_Te[cols]
Xte_Sat = csr_matrix(Sat_Ac_Te.values)

Sun_Ac = (Sun_Ac
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
Sun_Ac.head()
Sun_Ac.columns = ['device_id', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                  'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                  'h19', 'h20','h21', 'h22', 'h23', 'trainrow', 'testrow']
Sun_Ac.columns

Sun_Ac_Tr = Sun_Ac.copy(deep=True)
Sun_Ac_Tr.drop('device_id', inplace=True, axis=1)
Sun_Ac_Tr.drop('testrow', inplace=True, axis=1)
Sun_Ac_Tr = Sun_Ac_Tr.fillna(0)
Sun_Ac_Tr = Sun_Ac_Tr[Sun_Ac_Tr.trainrow != 0]
cols = Sun_Ac_Tr.columns.tolist()
cols = cols[-1:] + cols[:-1]
Sun_Ac_Tr = Sun_Ac_Tr[cols]
Xtr_Sun = csr_matrix(Sun_Ac_Tr.values)

Sun_Ac_Te = Sun_Ac.copy(deep=True)
Sun_Ac_Te.drop('device_id', inplace=True, axis=1)
Sun_Ac_Te.drop('trainrow', inplace=True, axis=1)
Sun_Ac_Te = Sun_Ac_Te.fillna(0)
Sun_Ac_Te = Sun_Ac_Te[Sun_Ac_Te.testrow != 0]
cols = Sun_Ac_Te.columns.tolist()
cols = cols[-1:] + cols[:-1]
Sun_Ac_Te = Sun_Ac_Te[cols]
Xte_Sun = csr_matrix(Sun_Ac_Te.values)

print('Mon Time Activity data: train shape {}, test shape {}'.format(Xtr_Mon.shape, Xte_Mon.shape))
print('Tue Time Activity data: train shape {}, test shape {}'.format(Xtr_Tue.shape, Xte_Tue.shape))
print('Wed Time Activity data: train shape {}, test shape {}'.format(Xtr_Wed.shape, Xte_Wed.shape))
print('Thu Time Activity data: train shape {}, test shape {}'.format(Xtr_Thu.shape, Xte_Thu.shape))
print('Fri Time Activity data: train shape {}, test shape {}'.format(Xtr_Fri.shape, Xte_Fri.shape))
print('Sat Time Activity data: train shape {}, test shape {}'.format(Xtr_Sat.shape, Xte_Sat.shape))
print('Sun Time Activity data: train shape {}, test shape {}'.format(Xtr_Sun.shape, Xte_Sun.shape))
#%% Device Activity per Week
print 'Device Activity Per Hour Per Week, Normalized'
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
#%% Creating Pickle objects per day
print 'Creating Pickle Objects per hour activity per week per device'
print 'By Error repeat previous command segment'

W17_Ac = (W17_Ac
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
W17_Ac.head()
W17_Ac.columns = ['device_id', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                  'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                  'h19', 'h20','h21', 'h22', 'h23', 'trainrow', 'testrow']
W17_Ac.columns

W17_Ac_Tr = W17_Ac.copy(deep=True)
W17_Ac_Tr.drop('device_id', inplace=True, axis=1)
W17_Ac_Tr.drop('testrow', inplace=True, axis=1)
W17_Ac_Tr = W17_Ac_Tr.fillna(0)
W17_Ac_Tr = W17_Ac_Tr[W17_Ac_Tr.trainrow != 0]
cols = W17_Ac_Tr.columns.tolist()
cols = cols[-1:] + cols[:-1]
W17_Ac_Tr = W17_Ac_Tr[cols]
Xtr_W17 = csr_matrix(W17_Ac_Tr.values)

W17_Ac_Te = W17_Ac.copy(deep=True)
W17_Ac_Te.drop('device_id', inplace=True, axis=1)
W17_Ac_Te.drop('trainrow', inplace=True, axis=1)
W17_Ac_Te = W17_Ac_Te.fillna(0)
W17_Ac_Te = W17_Ac_Te[W17_Ac_Te.testrow != 0]
cols = W17_Ac_Te.columns.tolist()
cols = cols[-1:] + cols[:-1]
W17_Ac_Te = W17_Ac_Te[cols]
Xte_W17 = csr_matrix(W17_Ac_Te.values)

W18_Ac = (W18_Ac
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
W18_Ac.head()
W18_Ac.columns = ['device_id', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                  'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                  'h19', 'h20','h21', 'h22', 'h23', 'trainrow', 'testrow']
W18_Ac.columns

W18_Ac_Tr = W18_Ac.copy(deep=True)
W18_Ac_Tr.drop('device_id', inplace=True, axis=1)
W18_Ac_Tr.drop('testrow', inplace=True, axis=1)
W18_Ac_Tr = W18_Ac_Tr.fillna(0)
W18_Ac_Tr = W18_Ac_Tr[W18_Ac_Tr.trainrow != 0]
cols = W18_Ac_Tr.columns.tolist()
cols = cols[-1:] + cols[:-1]
W18_Ac_Tr = W18_Ac_Tr[cols]
Xtr_W18 = csr_matrix(W18_Ac_Tr.values)

W18_Ac_Te = W18_Ac.copy(deep=True)
W18_Ac_Te.drop('device_id', inplace=True, axis=1)
W18_Ac_Te.drop('trainrow', inplace=True, axis=1)
W18_Ac_Te = W18_Ac_Te.fillna(0)
W18_Ac_Te = W18_Ac_Te[W18_Ac_Te.testrow != 0]
cols = W18_Ac_Te.columns.tolist()
cols = cols[-1:] + cols[:-1]
W18_Ac_Te = W18_Ac_Te[cols]
Xte_W18 = csr_matrix(W18_Ac_Te.values)

print('W17 Time Activity data: train shape {}, test shape {}'.format(Xtr_W17.shape, Xte_W17.shape))
print('W18 Time Activity data: train shape {}, test shape {}'.format(Xtr_W18.shape, Xte_W18.shape))

#%% Construct labels and nclasses
targetencoder = LabelEncoder().fit(gatrain.group)
train_labels = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)
label_encoding = targetencoder.inverse_transform(np.arange(nclasses))



#%% Save data

#Save train data
traindir = './Train_features/'
gendir = 'general/'
pickle.dump(Xtr_brand, open(traindir + 'phone_brand.p',"wb"))
pickle.dump(Xtr_model, open(traindir + 'phone_model.p',"wb"))
pickle.dump(Xtr_app, open(traindir + 'bool_apps_installed.p',"wb"))
pickle.dump(Xtr_label, open(traindir + 'bool_app_labels.p',"wb"))
pickle.dump(Xtr_day, open(traindir + 'phone_dayactive.p',"wb"))
pickle.dump(Xtr_hour, open(traindir + 'phone_houractive.p',"wb"))
pickle.dump(Xtr_lat, open(traindir + 'phone_lat.p',"wb"))
pickle.dump(Xtr_long, open(traindir + 'phone_long.p',"wb"))
pickle.dump(Xtr_weekday, open(traindir + 'phone_weekdayactive.p',"wb"))
pickle.dump(Xtr_Mon, open(traindir + 'phone_Mondayactive.p',"wb"))
pickle.dump(Xtr_Tue, open(traindir + 'phone_Tuesdayactive.p',"wb"))
pickle.dump(Xtr_Wed, open(traindir + 'phone_Wednesdayactive.p',"wb"))
pickle.dump(Xtr_Thu, open(traindir + 'phone_Thursdayactive.p',"wb"))
pickle.dump(Xtr_Fri, open(traindir + 'phone_Fridayactive.p',"wb"))
pickle.dump(Xtr_Sat, open(traindir + 'phone_Saturdayactive.p',"wb"))
pickle.dump(Xtr_Sun, open(traindir + 'phone_Sundayactive.p',"wb"))
pickle.dump(Xtr_W17, open(traindir + 'phone_W17active.p',"wb"))
pickle.dump(Xtr_W18, open(traindir + 'phone_W18active.p',"wb"))

pickle.dump(train_labels, open(traindir + gendir + 'labels.p',"wb"))
pickle.dump(nclasses, open(traindir + gendir + 'nclasses.p',"wb"))
pickle.dump(label_encoding, open(traindir + gendir + 'label_encoding.p',"wb"))

#Save test data
testdir = './Test_features/'
pickle.dump(Xte_brand, open(testdir + 'phone_brand.p',"wb"))
pickle.dump(Xte_model, open(testdir + 'phone_model.p',"wb"))
pickle.dump(Xte_app, open(testdir + 'bool_apps_installed.p',"wb"))
pickle.dump(Xte_label, open(testdir + 'bool_app_labels.p',"wb"))
pickle.dump(Xte_day, open(testdir + 'phone_dayactive.p',"wb"))
pickle.dump(Xte_hour, open(testdir + 'phone_houractive.p',"wb"))
pickle.dump(Xte_lat, open(testdir + 'phone_lat.p',"wb"))
pickle.dump(Xte_long, open(testdir + 'phone_long.p',"wb"))
pickle.dump(Xte_weekday, open(testdir + 'phone_weekdayactive.p',"wb"))
pickle.dump(Xte_Mon, open(traindir + 'phone_Mondayactive.p',"wb"))
pickle.dump(Xte_Tue, open(traindir + 'phone_Tuesdayactive.p',"wb"))
pickle.dump(Xte_Wed, open(traindir + 'phone_Wednesdayactive.p',"wb"))
pickle.dump(Xte_Thu, open(traindir + 'phone_Thursdayactive.p',"wb"))
pickle.dump(Xte_Fri, open(traindir + 'phone_Fridayactive.p',"wb"))
pickle.dump(Xte_Sat, open(traindir + 'phone_Saturdayactive.p',"wb"))
pickle.dump(Xte_Sun, open(traindir + 'phone_Sundayactive.p',"wb"))
pickle.dump(Xte_W17, open(traindir + 'phone_W17active.p',"wb"))
pickle.dump(Xte_W18, open(traindir + 'phone_W18active.p',"wb"))
