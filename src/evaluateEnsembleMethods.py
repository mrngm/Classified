import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import datetime

def eval_rmsle(val_prices,pred_prices):
    return np.sqrt(np.mean(np.square(np.subtract(np.log(np.add(pred_prices,1)),np.log(np.add(val_prices,1))))))
    

print('\nWelcome to "duurt laang" ETL services. \n\n..Reading in training data as X_train..')

X_train = pd.read_csv('../data/train_2014.csv')
y_train = X_train['price_doc'].values
X_train.drop(['price_doc'], axis=1, inplace=True)

print('..Loading test set as X_test')
X_test = pd.read_csv('../data/test_2015.csv')
ids = X_test['id']


print('..Fetching true class labels')
y_test = pd.read_csv('../data/val_prices.csv')
print(y_test.shape)
print(y_test['price_doc'].values)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(ids.shape)

# Random Forest 6.12138, if we take just the data from 2014 we get 6.12448. A lower score indicating better performance. 
print('..RandomForestRegressor with n = 10 ..')
clf = RandomForestRegressor(n_estimators=30)
clf.fit(X_train, y_train)
y_rf = clf.predict(X_test)
print y_rf.shape

#evaluate
print eval_rmsle(y_test['price_doc'].values, y_rf)

#construct a sample submission for ensembling, per xgb-baseline2
time = datetime.datetime.now().strftime("%A, %d %B %Y %I%M%p")
temp = pd.DataFrame({'id': ids, 'price_doc': y_rf})
temp.head()
temp.to_csv('../output/RF_' + time + '.csv', index=False)

# Gradient Boosting Regressor 
print('..Gradient Boosting ..')
gb = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2, learning_rate=0.01)
gb.fit(X_train, y_train)
y_gb = gb.predict(X_test)
print y_gb.shape

#evaluate
print eval_rmsle(y_test['price_doc'].values, y_gb)

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

#evaluate
print eval_rmsle(y_test['price_doc'].values, y_ad)

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

#evaluate
print eval_rmsle(y_test['price_doc'].values, y_ba)

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

#evaluate
print eval_rmsle(y_test['price_doc'].values, y_ex)

#construct a sample submission for ensembling, per xgb-baseline2
time = datetime.datetime.now().strftime("%A, %d %B %Y %I%M%p")
temp = pd.DataFrame({'id': ids, 'price_doc': y_ex})
temp.head()
temp.to_csv('../output/extraTrees_' + time + '.csv', index=False)