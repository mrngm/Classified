# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 17:39:11 2017

@author: Gebruiker
"""

from sklearn.metrics import mean_squared_error
import math
from math import sqrt



y_pred = map(math.log10, y_pred)
X_test = map(math.log10, X_test)

rms = sqrt(mean_squared_error(X_test, y_pred))
print rms

