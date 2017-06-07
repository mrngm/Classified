# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 16:21:16 2017

@author: Roel
"""

import numpy as np

def eval_rmsle(val_prices,pred_prices):
    return np.sqrt(np.mean(np.square(np.subtract(np.log(np.add(pred_prices,1)),np.log(np.add(val_prices,1))))))
    