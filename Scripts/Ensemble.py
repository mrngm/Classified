# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:25:10 2017

@author: Gebruiker
"""


import numpy as np

def eval_rmsle(val_prices,pred_prices):
    return np.sqrt(np.mean(np.square(np.subtract(np.log(np.add(pred_prices,1)),np.log(np.add(val_prices,1))))))