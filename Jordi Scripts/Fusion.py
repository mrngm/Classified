# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 14:08:27 2017

@author: Gebruiker
"""

from __future__ import division
"""
Programming Preparation
"""

import numpy as np
import pandas as pd
import datetime as dt
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
"""
Data Structure
"""
def _parse_date(date_str, format_str):
    time_dt = dt.datetime.strptime(date_str, format_str)
    return [time_dt.year, time_dt.month, time_dt.day, time_dt.weekday, time_dt.time()]


gender_train = pd.read_csv('D:\School\School\Master\Jaar_1\Machine Learning in Practice\Competition\Data\Mobile Data\gender_age_train.csv')
app_events = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/app_events.csv')
events = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/events.csv')
label_cat = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/label_categories.csv')
app_labels = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/app_labels.csv')
brand_model = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Mobile Data/phone_brand_device_model.csv')

gender_train.describe()
app_events.describe()
events.describe()
label_cat.describe()
brand_model.describe()
app_labels.describe()

events.columns
gender_train.columns
app_events.columns
label_cat.columns
brand_model.columns
app_labels.columns

brand_model.replace('三星', 'SAMSUNG', inplace=True)
brand_model.replace('华为', 'HUAWEI', inplace=True)
brand_model.replace('小米', 'XIAOMI', inplace=True)
brand_model.replace('vivo', 'VIVO', inplace=True)
brand_model.replace('魅族', 'MEIZU', inplace=True)
brand_model.replace('酷派', 'NOKIA', inplace=True)
brand_model.replace('天语', 'SAMSUNG-JITTERBUG', inplace=True)
brand_model.replace('金立', 'GIONEE', inplace=True)
brand_model.replace('联想', 'LENOVO', inplace=True)
brand_model.replace('海信', 'HISENSE', inplace=True)
brand_model.replace('索尼', 'SONY', inplace=True)
brand_model.replace('酷比', 'SAMSUNG-GALAXY', inplace=True)
brand_model.replace('中兴', 'LG', inplace=True)
brand_model.replace('康佳', 'ZTE', inplace=True)
brand_model.replace('旗舰版', 'Ultimate', inplace=True)
brand_model.replace('奇酷', 'QI-Cool', inplace=True)
brand_model.replace('欧博信', 'OPPSON', inplace=True)
brand_model.replace('欧比', 'LG', inplace=True)
brand_model.replace('爱派尔', 'CHINESE', inplace=True)
brand_model.replace('努比亚', 'SONY-EXPERIA', inplace=True)
brand_model.replace('优米', 'XIAOMI', inplace=True)
brand_model.replace('朵唯', 'SAMSUNG-GALAXY', inplace=True)
brand_model.replace('黑米', 'GEOTEL', inplace=True)
brand_model.replace('努比亚', 'ZTE-NUBIA', inplace=True)
brand_model.replace('锤子', 'SMARTISAN', inplace=True)
brand_model.replace('朵唯', 'HYUNDAI', inplace=True)
brand_model.replace('酷比魔方', 'CUBE-TALK', inplace=True)
brand_model.replace('E人E本', 'PANASONIC', inplace=True)
brand_model.replace('E派', 'TELSTRA', inplace=True)
brand_model.replace('一加', 'NOKIA', inplace=True)
brand_model.replace('世纪天元', 'Century-Tianyuan', inplace=True)
brand_model.replace('中国移动', 'ZONG', inplace=True)
brand_model.replace('世纪星', 'INTEX-AQUA', inplace=True)
brand_model.replace('乐视', 'LE-PRO', inplace=True)  
brand_model.replace('乡米', 'XMKJ', inplace=True)  
brand_model.replace('亚马逊', 'AMAZON-FIRE PHONE', inplace=True)  
brand_model.replace('亿通', 'CHINESE', inplace=True) 
brand_model.replace('优语', 'KDOOR', inplace=True) 
brand_model.replace('优购', 'CHINESE', inplace=True) 
brand_model.replace('先锋', 'PIONEER', inplace=True) 
brand_model.replace('凯利通', 'PANASONIC', inplace=True) 
brand_model.replace('华硕', 'ASUS', inplace=True) 
brand_model.replace('原点', 'ORIGINAL', inplace=True) 
brand_model.replace('台电', 'TECLAST', inplace=True) 
brand_model.replace('唯米', 'HUAWEI', inplace=True) 
brand_model.replace('基伍', 'BAIMI', inplace=True) 
brand_model.replace('夏新', 'AMOI', inplace=True) 
brand_model.replace('大Q', 'CHINESE', inplace=True)
brand_model.replace('大可乐', 'DAKELE', inplace=True)
brand_model.replace('大显', 'HUAWEI-HONOR', inplace=True)
brand_model.replace('天宏时代', 'TIANHONG-TIMES', inplace=True)
brand_model.replace('奥克斯', 'AUX', inplace=True)
brand_model.replace('宏碁', 'ACER', inplace=True)
brand_model.replace('宏碁', 'BASICOM', inplace=True)
brand_model.replace('富可视', 'VOXTEL', inplace=True)
brand_model.replace('小杨树', 'SHENZHEN-XIAOYANGSHU-COMMUNICATION', inplace=True)
brand_model.replace('尼比鲁', 'NIBURU', inplace=True)
brand_model.replace('帷幄', 'LAVA-IRIS', inplace=True)
brand_model.replace('广信', 'ADVAN-BARCA-5', inplace=True)
brand_model.replace('德卡诺', 'LANDVO', inplace=True)
brand_model.replace('德赛', 'DESAY', inplace=True)
brand_model.replace('恒宇丰', 'KOU-FENG-HYF', inplace=True)
brand_model.replace('惠普', 'HP', inplace=True)
brand_model.replace('戴尔', 'DELL', inplace=True)
brand_model.replace('摩托罗拉', 'LENOVO', inplace=True)
brand_model.replace('斐讯', 'PHICOMM', inplace=True)
brand_model.replace('昂达', 'ONDA', inplace=True)
brand_model.replace('易派', 'SONY-XPERIA', inplace=True)
brand_model.replace('普耐尔', 'PLOYER', inplace=True)
brand_model.replace('智镁', 'ROLAND', inplace=True)
brand_model.replace('智镁', 'VIVO', inplace=True)
brand_model.replace('极米', 'i-STATION', inplace=True)
brand_model.replace('果米', 'SAMSUNG-GALAXY', inplace=True)
brand_model.replace('梦米', 'HUAWEI-ASCEND', inplace=True)
brand_model.replace('欧乐迪', 'i-MOBILE', inplace=True)
brand_model.replace('欧乐酷', 'CBSKY-UWATCH', inplace=True)
brand_model.replace('欧奇', 'OU QI', inplace=True)
brand_model.replace('沃普丰', 'SAMSUNG-GALAXY', inplace=True)
brand_model.replace('波导', 'HCR', inplace=True)
brand_model.replace('海尔', 'HAIER', inplace=True)
brand_model.replace('瑞米', 'VIVO', inplace=True)
brand_model.replace('瑞高', 'MUCH', inplace=True)
brand_model.replace('白米', 'HKS', inplace=True)
brand_model.replace('百加', 'X-BO', inplace=True)
brand_model.replace('百立丰', 'LI-FENG', inplace=True)
brand_model.replace('神舟', 'HASEE', inplace=True)
brand_model.replace('米奇', 'GAMMA', inplace=True)
brand_model.replace('米歌', 'SEAGULL', inplace=True)
brand_model.replace('糖葫芦', 'THL', inplace=True)
brand_model.replace('糯米', 'XIAOMI-REDMI', inplace=True)
brand_model.replace('纽曼', 'NEWMAN', inplace=True)
brand_model.replace('维图', 'ORCELL', inplace=True)
brand_model.replace('美图', 'MEITU', inplace=True)
brand_model.replace('聆韵', 'PATENT', inplace=True)
brand_model.replace('至尊宝', 'CHINA-QUADBAND-TV', inplace=True)
brand_model.replace('艾优尼', 'SAMSUNG', inplace=True)
brand_model.replace('蓝魔', 'RAMOS', inplace=True)
brand_model.replace('虾米', 'VIVO', inplace=True)
brand_model.replace('西米', 'XIMI-SIMI', inplace=True)
brand_model.replace('西门子', 'GIGASET-ME', inplace=True)
brand_model.replace('语信', 'SAMSUNG-GALAXY', inplace=True)
brand_model.replace('诺亚信', 'MAXIMUM', inplace=True)
brand_model.replace('诺基亚', 'OPPO', inplace=True)
brand_model.replace('谷歌', 'ASUS-GOOGLE', inplace=True)
brand_model.replace('贝尔丰', 'T-SERIES', inplace=True)
brand_model.replace('赛博宇华', 'SAIBOYUHUA', inplace=True)
brand_model.replace('邦华', 'OUKITEL', inplace=True)
brand_model.replace('酷珀', 'KOPO', inplace=True)
brand_model.replace('金星数码', 'VENUS-JXD', inplace=True)
brand_model.replace('长虹', 'GABA', inplace=True)
brand_model.replace('青橙', 'GREEN-ORANGE', inplace=True)
brand_model.replace('青葱', 'LAVA', inplace=True)
brand_model.replace('飞利浦', 'PHILIPS-XENIUM', inplace=True)
brand_model.replace('飞秒', 'PANASONIC', inplace=True)
brand_model.replace('首云', 'PEPSI', inplace=True)
brand_model.replace('鲜米', 'HI-TECH', inplace=True)

#Mobile phone coupled with Gender
Mobile_Phone_gender = pd.merge(left=gender_train,right=brand_model, left_on='device_id', right_on='device_id')
Mobile_Phone_gender.shape
print Mobile_Phone_gender

Mobile_Phone_gender.to_csv('Phone_gender.csv')
Phone_gender = pd.read_csv('C:/Users/Gebruiker/Phone_gender.csv', index_col = 0)
Phone_gender.describe()
Phone_gender.columns

#Apps coupled with labels
Checking_Apps = pd.merge(left=app_labels, right=label_cat, left_on='label_id', right_on='label_id')
Checking_Apps.shape
print Checking_Apps

Checking_Apps.to_csv('Check_Apps.csv')
Check_Apps = pd.read_csv('C:/Users/Gebruiker/Check_Apps.csv', index_col = 0)
Check_Apps.describe()
Check_Apps.columns


#Geolocation gender and phones
Mobile_Phone_gender_location = pd.merge(left=Phone_gender,right=events, left_on='device_id', right_on='device_id')
Mobile_Phone_gender_location.shape
print Mobile_Phone_gender_location

Mobile_Phone_gender_location = Mobile_Phone_gender_location[Mobile_Phone_gender_location.longitude != 0.00]
Mobile_Phone_gender_location.shape
print Mobile_Phone_gender_location

Mobile_Phone_gender_location = Mobile_Phone_gender_location[Mobile_Phone_gender_location.latitude != 0.00]
Mobile_Phone_gender_location.shape
print Mobile_Phone_gender_location

Mobile_Phone_gender_location.to_csv('Phone_Gender_Geo.csv')
Phone_Gender_Geo = pd.read_csv('C:/Users/Gebruiker/Phone_Gender_Geo.csv', index_col = 0)
Phone_Gender_Geo.describe()
Phone_Gender_Geo.columns

#Controlling Apps on Events
Checking_Apps_Events = pd.merge(left=Check_Apps,right=app_events, left_on='app_id', right_on='app_id')
Checking_Apps_Events.to_csv('Checking_Apps_Events.csv')

#Universal Database
Phone_gender_geo = pd.read_csv('C:/Users/Gebruiker/Phone_Gender_Geo.csv', index_col = 0)
Phone_gender_geo.columns
Phone_gender_geo.head(10)
Events = pd.read_csv('C:/Users/Gebruiker/Checking_Apps_Events.csv', index_col = 0)
B = pd.merge(left=Phone_gender_geo,right=app_events, left_on='event_id', right_on='event_id')
B.to_csv('B.csv')

B = pd.read_csv('C:/Users/Gebruiker/B.csv', index_col = 0)
Universal = pd.merge(left=B,right=Check_Apps, left_on='app_id', right_on='app_id')
Universal.describe()
Universal.columns
Universal.to_csv('Universal.csv')