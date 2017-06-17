# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:34:37 2017

@author: Gebruiker
"""
# Parameters
micro_humility_factor = 1     #    range from 0 (complete humility) to 1 (no humility)
macro_humility_factor = 0.96
jason_weight = .2
bruno_weight = .2
reynaldo_weight = 1 - jason_weight - bruno_weight

# Get ready for lots of annoying deprecation warnings
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
import scipy as sp

# Functions to use in data adjustment

def scale_miss(   # Scale shifted logs and compare raw stdev to old raw stdev
        alpha,
        shifted_logs,
        oldstd,
        new_logmean
        ):
    newlogs = new_logmean + alpha*(shifted_logs - new_logmean)
    newstd = np.std(np.exp(newlogs))
    return (newstd-oldstd)**2
    

def shift_logmean_but_keep_scale(  # Or change the scale, but relative to the old scale
        data,
        new_logmean,
        rescaler
        ):
    logdata = np.log(data)
    oldstd = data.std()
    shift = new_logmean - logdata.mean()
    shifted_logs = logdata + shift
    scale = sp.optimize.leastsq( scale_miss, 1, args=(shifted_logs, oldstd, new_logmean) )
    alpha = scale[0][0]
    newlogs = new_logmean + rescaler*alpha*(shifted_logs - new_logmean)
    return np.exp(newlogs)

# Read data
macro = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/macro.csv')
train = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/train.csv')
test = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/test.csv')

# Macro data monthly medians
macro["timestamp"] = pd.to_datetime(macro["timestamp"])
macro["year"]  = macro["timestamp"].dt.year
macro["month"] = macro["timestamp"].dt.month
macro["yearmonth"] = 100*macro.year + macro.month
macmeds = macro.groupby("yearmonth").median()

# Price data monthly medians
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["year"]  = train["timestamp"].dt.year
train["month"] = train["timestamp"].dt.month
train["yearmonth"] = 100*train.year + train.month
prices = train[["yearmonth","price_doc"]]
p = prices.groupby("yearmonth").median()

# Join monthly prices to macro data
df = macmeds.join(p)

# Function to process Almon lags

import numpy.matlib as ml
 
def almonZmatrix(X, maxlag, maxdeg):
    """
    Creates the Z matrix corresponding to vector X.
    """
    n = len(X)
    Z = ml.zeros((len(X)-maxlag, maxdeg+1))
    for t in range(maxlag,  n):
       #Solve for Z[t][0].
       Z[t-maxlag,0] = sum([X[t-lag] for lag in range(maxlag+1)])
       for j in range(1, maxdeg+1):
             s = 0.0
             for i in range(1, maxlag+1):       
                s += (i)**j * X[t-i]
             Z[t-maxlag,j] = s
    return Z

# Prepare data for macro model
y = df.price_doc.div(df.cpi).apply(np.log).loc[201108:201506]
lncpi = df.cpi.apply(np.log)
tblags = 5    # Number of lags used on PDL for Trade Balance
mrlags = 5    # Number of lags used on PDL for Mortgage Rate
cplags = 5    # Number of lags used on PDL for CPI
ztb = almonZmatrix(df.balance_trade.loc[201103:201506].as_matrix(), tblags, 1)
zmr = almonZmatrix(df.mortgage_rate.loc[201103:201506].as_matrix(), mrlags, 1)
zcp = almonZmatrix(lncpi.loc[201103:201506].as_matrix(), cplags, 1)
columns = ['tb0', 'tb1', 'mr0', 'mr1', 'cp0', 'cp1']
z = pd.DataFrame( np.concatenate( (ztb, zmr, zcp), axis=1), y.index.values, columns )
X = sm.add_constant( z )

# Fit macro model
eq = sm.OLS(y, X)
fit = eq.fit()

# Predict with macro model
test_cpi = df.cpi.loc[201507:201605]
test_index = test_cpi.index
ztb_test = almonZmatrix(df.balance_trade.loc[201502:201605].as_matrix(), tblags, 1)
zmr_test = almonZmatrix(df.mortgage_rate.loc[201502:201605].as_matrix(), mrlags, 1)
zcp_test = almonZmatrix(lncpi.loc[201502:201605].as_matrix(), cplags, 1)
z_test = pd.DataFrame( np.concatenate( (ztb_test, zmr_test, zcp_test), axis=1), 
                       test_index, columns )
X_test = sm.add_constant( z_test )
pred_lnrp = fit.predict( X_test )
pred_p = np.exp(pred_lnrp) * test_cpi

# Merge with test cases and compute mean for macro prediction
test["timestamp"] = pd.to_datetime(test["timestamp"])
test["year"]  = test["timestamp"].dt.year
test["month"] = test["timestamp"].dt.month
test["yearmonth"] = 100*test.year + test.month
test_ids = test[["yearmonth","id"]]
monthprices = pd.DataFrame({"yearmonth":pred_p.index.values,"monthprice":pred_p.values})
macro_mean = np.exp(test_ids.merge(monthprices, on="yearmonth").monthprice.apply(np.log).mean())
macro_mean

# Naive macro model assumes housing prices will simply follow CPI
naive_pred_lnrp = y.mean()
naive_pred_p = np.exp(naive_pred_lnrp) * test_cpi
monthnaive = pd.DataFrame({"yearmonth":pred_p.index.values, "monthprice":naive_pred_p.values})
macro_naive = np.exp(test_ids.merge(monthnaive, on="yearmonth").monthprice.apply(np.log).mean())
macro_naive

# Combine naive and substantive macro models
macro_mean = macro_naive * (macro_mean/macro_naive) ** macro_humility_factor
macro_mean

# Bruno with outlier dropped

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/Validatieset/train_2014.csv", parse_dates=['timestamp'])
df_test = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/Validatieset/test_2015.csv', parse_dates=['timestamp'])
df_macro = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/macro.csv", parse_dates=['timestamp'])
df_test = df_test.drop('Unnamed: 0', 1)
train = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/train.csv')

#Split on instances before(<) and in(>=) 2015
train_subset = train[train['id'] < 27235]
train_2014 = train_subset[train_subset['id'] >13572]
validation = train[train['id'] >= 27235]

val_prices = validation['price_doc']
y_test = val_prices
validation = validation.drop('price_doc', 1)



df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

y_train = df_train['price_doc'].values
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
# Next line just adds a lot of NA columns (becuase "join" only works on indexes)
# but somewhow it seems to affect the result
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)


factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)


num_boost_round = 489  # From Bruno's original CV, I think
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)

y_predict = model.predict(dtest)
bruno_model_raw_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
bruno_model_raw_output.head()
np.exp( bruno_model_raw_output.price_doc.apply(np.log).mean() )
# Adjust
lnm = np.log(macro_mean)
y_predict = shift_logmean_but_keep_scale( y_predict, lnm, micro_humility_factor )

bruno_model_adjusted_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
bruno_model_adjusted_output.head() 

#%%
def eval_rmsle(y_test,score):
    return np.sqrt(np.mean(np.square(np.subtract(np.log(np.add(y_predict,1)),np.log(np.add(y_test,1))))))

RMSLE = np.sqrt(np.mean(np.square(np.subtract(np.log(np.add(y_predict,1)),np.log(np.add(y_test,1))))))
print RMSLE