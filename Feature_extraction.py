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
events = events.assign(
    date_year = lambda d: d['timestamp'].dt.year,
    date_month = lambda d: d['timestamp'].dt.month,
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
#%% Device Distance
distance = distance.assign(
    date_year = lambda d: d['timestamp'].dt.year,
    date_month = lambda d: d['timestamp'].dt.month,
    date_day = lambda d: d['timestamp'].dt.day,
    date_hour = lambda d: d['timestamp'].dt.hour,
    date_minute = lambda d: d['timestamp'].dt.minute,
    date_second = lambda d: d['timestamp'].dt.second,
    date_weekday = lambda d: d['timestamp'].dt.weekday,
)

distance_events = pd.DataFrame(columns=["device_id", "date", "day_distance"])

distance.sort_values(['device_id', 'timestamp'], ascending=[True, True], inplace=True)

per_device = distance.groupby('device_id')

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

distancecoder = LabelEncoder().fit(distance_events.day_distance)
distance_events['day_distance'] = distancecoder.transform(distance_events.day_distance)
ndistance = len(distancecoder.classes_)
devicedistance = (distance_events[['device_id','day_distance']]
                .groupby(['device_id','day_distance'])['day_distance'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicedistance.head()

d = devicedistance.dropna(subset=['trainrow'])
Xtr_distance = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.day_distance)), 
                      shape=(gatrain.shape[0],ndistance))
d = devicedistance.dropna(subset=['testrow'])
Xte_distance = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.day_distance)), 
                      shape=(gatest.shape[0],ndistance))
print('Device Distance: train shape {}, test shape {}'.format(Xtr_distance.shape, Xte_distance.shape))

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
pickle.dump(Xtr_distance, open(traindir + 'phone_distance.p',"wb"))
pickle.dump(Xtr_hour, open(traindir + 'phone_houractive.p',"wb"))
pickle.dump(Xtr_lat, open(traindir + 'phone_lat.p',"wb"))
pickle.dump(Xtr_long, open(traindir + 'phone_long.p',"wb"))
pickle.dump(Xtr_weekday, open(traindir + 'phone_weekdayactive.p',"wb"))

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
pickle.dump(Xte_distance, open(testdir + 'phone_distance.p',"wb"))
pickle.dump(Xte_hour, open(testdir + 'phone_houractive.p',"wb"))
pickle.dump(Xte_lat, open(testdir + 'phone_lat.p',"wb"))
pickle.dump(Xte_long, open(testdir + 'phone_long.p',"wb"))
pickle.dump(Xte_weekday, open(testdir + 'phone_weekdayactive.p',"wb"))
