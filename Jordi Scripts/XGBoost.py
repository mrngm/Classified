# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 16:45:39 2017

@author: Gebruiker
"""

#%% Programming Library
print 'library import'
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

#%% Importing Dataset
print 'importing dataset'
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

#%% Encoding data
print 'Feature Extraction'
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))

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

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())

d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))

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
#%% Time Based
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
#%%
print('Day data: train shape {}, test shape {}'.format(Xtr_day.shape, Xte_day.shape))
print('Hour data: train shape {}, test shape {}'.format(Xtr_hour.shape, Xte_hour.shape))
print('Weekday data: train shape {}, test shape {}'.format(Xtr_weekday.shape, Xte_weekday.shape))
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
#%% Combining the Xtr and Xte
print 'Combine'
Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_hour, Xtr_day, Xtr_weekday), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_hour, Xte_day, Xte_weekday), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
#%% XGBoost
print 'XGBoost Performance'

params = {}
params['booster'] = 'gblinear'
params['objective'] = "multi:softprob"
params['eval_metric'] = 'mlogloss'
params['eta'] = 0.005
params['num_class'] = 12
params['lambda'] = 3
params['alpha'] = 1

# Random 10% for validation
kf = list(StratifiedKFold(y, n_folds=10, shuffle=True, random_state=4242))[0]

Xtr, Xte = Xtrain[kf[0], :], Xtrain[kf[1], :]
ytr, yte = y[kf[0]], y[kf[1]]

print('Training set: ' + str(Xtr.shape))
print('Validation set: ' + str(Xte.shape))

d_train = xgb.DMatrix(Xtr, label=ytr)
d_valid = xgb.DMatrix(Xte, label=yte)

watchlist = [(d_train, 'train'), (d_valid, 'eval')]

clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=25)

pred = clf.predict(xgb.DMatrix(Xtest))

pred = pd.DataFrame(pred, index = gatest.index, columns=targetencoder.classes_)
pred.head()
pred.to_csv('sparse_xgb.csv', index=True)

#params['lambda'] = 1
#for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#    params['alpha'] = alpha
#    clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=25)
#    print('^' + str(alpha))