# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 14:00:31 2017

@author: Gebruiker
"""
#%% Library
print 'Importing Data and library'
import pandas as pd
import os
import numpy as np

datadir = 'D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data'
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first')
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     parse_dates=['timestamp'], index_col='event_id')

#%% Filtering data
print 'filtering device_id based on apps'

P_unique_devices = np.unique(phone['device_id'])
E_unique_devices = np.unique(events['device_id'])
P_unique_devices = P_unique_devices.tolist()
E_unique_devices = E_unique_devices.tolist()

unique_devices = [item for item in P_unique_devices if item not in E_unique_devices]
#%%
for u in unique_devices[:5]:
    device_data = events[events['device_id']==u]
    
    events_csv = open('./device_data_files/'+unicode(u)+'.txt', 'w')
    device_data.to_csv(events_csv, sep = ' ', header=False, index=False, 
                       columns['longitude'], encoding = 'utf-8')
    data_csv.close()
    
