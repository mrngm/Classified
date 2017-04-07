# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 14:53:09 2017

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

Check_Apps = pd.read_csv('C:/Users/Gebruiker/Check_Apps.csv', index_col = 0)
app_events = pd.read_csv('D:/School/School/Master/Jaar 1/Machine Learning in Practice/Competition/Data/Mobile Data/app_events.csv')

Checking_Apps_Events = pd.merge(left=Check_Apps,right=app_events, how='right', left_on='app_id', right_on='app_id')
Checking_Apps_Events.to_csv('Checking_Apps_Events.csv')
