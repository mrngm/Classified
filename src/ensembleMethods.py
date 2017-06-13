import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import datetime

print('\nWelcome to "duurt laang" ETL services. \n\n..Reading in training data as X_train..')
X_train = pd.read_csv('../data/one-hot_median_filled_train.csv')

print('..Loading test set as X_test')
tX_test = pd.read_csv('../data/one-hot_median_filled_test.csv')
ids = tX_test['id']
X_test = tX_test

print('..Fetching true class labels')
ty_train = pd.read_csv('../data/train_prices.csv')
y_train = ty_train['price_doc'].values
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(ids.shape)

# Random Forest 
print('..RandomForestRegressor with n = 10 ..')
clf = RandomForestRegressor(n_estimators=10)
clf.fit(X_train, y_train)
y_rf = clf.predict(X_test)
print y_rf.shape

#construct a sample submission for ensembling, per xgb-baseline2
time = datetime.datetime.now().strftime("%A, %d %B %Y %I%M%p")
temp = pd.DataFrame({'id': ids, 'price_doc': y_rf})
temp.head()
temp.to_csv('../output/RF_' + time + '.csv', index=False)

# Gradient Boosting Regressor 
print('..Gradient Boosting ..')
gb = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2, learning_rate=0.01)
gb.fit(X_train, y_train)
y_gb = clf.predict(X_test)
print y_gb.shape

#construct a sample submission for ensembling, per xgb-baseline2
time = datetime.datetime.now().strftime("%A, %d %B %Y %I%M%p")
temp = pd.DataFrame({'id': ids, 'price_doc': y_gb})
temp.head()
temp.to_csv('../output/GB_' + time + '.csv', index=False)

# Adaboost Regressor
print('..Adaboost with 300 estimators..')
rng = np.random.RandomState(1)
ad = AdaBoostRegressor(n_estimators=300, random_state=rng)
ad.fit(X_train, y_train)
y_ad = ad.predict(X_test)
print y_ad.shape

#construct a sample submission for ensembling, per xgb-baseline2
time = datetime.datetime.now().strftime("%A, %d %B %Y %I%M%p")
temp = pd.DataFrame({'id': ids, 'price_doc': y_ad})
temp.head()
temp.to_csv('../output/ADA_' + time + '.csv', index=False)

# Bagging Regressor 
print('..Bagging regressor with 300 estimators..')
rng = np.random.RandomState(1)
ba = BaggingRegressor(n_estimators=300, random_state=rng)
ba.fit(X_train, y_train)
y_ba = ba.predict(X_test)
print y_ba.shape

#construct a sample submission for ensembling, per xgb-baseline2
time = datetime.datetime.now().strftime("%A, %d %B %Y %I%M%p")
temp = pd.DataFrame({'id': ids, 'price_doc': y_ba})
temp.head()
temp.to_csv('../output/Bag_' + time + '.csv', index=False)

# Extra Trees Regressor
print('..Extra Trees Regressor with n=10..')
rng = np.random.RandomState(1)
ex = ExtraTreesRegressor(n_estimators=10, max_features="auto", random_state=rng)
ex.fit(X_train, y_train)
y_ex = ex.predict(X_test)
print y_ex.shape

#construct a sample submission for ensembling, per xgb-baseline2
time = datetime.datetime.now().strftime("%A, %d %B %Y %I%M%p")
temp = pd.DataFrame({'id': ids, 'price_doc': y_ex})
temp.head()
temp.to_csv('../output/extraTrees_' + time + '.csv', index=False)