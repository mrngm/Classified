# -*- coding: utf-8 -*-
"""
Created on Sun May 07 14:31:31 2017

@author: Gebruiker
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()
import datetime
import matplotlib.dates as mdates

%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)
#%% Importing Train

print 'importing'
train_df = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/train.csv")
train_df.shape
print 'head Train database'
train_df.head()
#%% Importing Macro
print 'importing Macro'
macro_df = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/macro.csv")
macro_df.shape
print 'head Macro database'
macro_df.head()
#%% Joining

print "Joining information"

house_market_df = pd.merge(left=train_df,right=macro_df, how='left', left_on='timestamp', right_on='timestamp')
print "Data structure"
house_market_df.head()
house_market_df.shape
house_market_df.columns

#%% First ten rows
house_market_df.head(10)
#%%
train_df.shape
#%%
macro_df.shape
#%% Cleaning data
train_df = train_df[train_df.ecology != 'no data']
#%% Columns

print 'Columns'
column = train_df.columns
print column
#%% Exploration

print 'Plotting graph'
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot(train_df.price_doc.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot(np.log(train_df.price_doc.values), bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()

print 'Median House Price'

train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Year Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

print 'Minimum Area House Price'

grouped_df = train_df.groupby('sub_area')['price_doc'].aggregate(np.max).reset_index()
grouped_df = grouped_df.sort(['price_doc']).reset_index(drop=True)
plt.figure(figsize=(25,8))
sns.barplot(grouped_df.sub_area.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Maximum Price', fontsize=12)
plt.xlabel('Sub_Area', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

print 'Median Build Year House Price'

groeped_df = train_df.groupby('build_year')['price_doc'].aggregate(np.median).reset_index()
groeped_df = groeped_df.sort(['price_doc']).reset_index(drop=True)
plt.figure(figsize=(25,8))
sns.barplot(groeped_df.build_year.values, groeped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Build Year', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

print 'Median Build Year Full SQ'
house_market_df['price_per_sqm']=house_market_df['price_doc']/house_market_df['full_sq']
groeped_df = house_market_df.groupby('sub_area')['price_per_sqm'].aggregate(np.median).reset_index()
groeped_df = groeped_df.sort(['price_per_sqm']).reset_index(drop=True)
plt.figure(figsize=(25,8))
sns.barplot(groeped_df.sub_area.values, groeped_df.price_per_sqm.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price Per SQM', fontsize=12)
plt.xlabel('Area', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

print 'Median Full Square per region'

groeped_df = train_df.groupby('sub_area')['full_sq'].aggregate(np.median).reset_index()
groeped_df = groeped_df.sort(['full_sq']).reset_index(drop=True)
plt.figure(figsize=(25,8))
sns.barplot(groeped_df.sub_area.values, groeped_df.full_sq.values, alpha=0.8, color=color[2])
plt.ylabel('Median Space', fontsize=12)
plt.xlabel('Area', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

print 'Apartment Prices'

groeped_df = house_market_df.groupby('apartment_build')['price_doc'].aggregate(np.median).reset_index()
groeped_df = groeped_df.sort(['price_doc']).reset_index(drop=True)
plt.figure(figsize=(25,8))
sns.barplot(groeped_df.apartment_build.values, groeped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median prices', fontsize=12)
plt.xlabel('Aparment', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

groeped_df = train_df.groupby('ecology')['price_doc'].aggregate(np.max).reset_index()
groeped_df = groeped_df.sort(['price_doc']).reset_index(drop=True)
plt.figure(figsize=(25,8))
sns.barplot(groeped_df.ecology.values, groeped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Max Price', fontsize=12)
plt.xlabel('Ecology', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

#%% Property Information
print 'Property information'

train_df[train_df.price_doc.isin([1e6, 2e6, 3e6, 4e6, 5e6])].product_type.value_counts()
train_df[~train_df.price_doc.isin([1e6, 2e6, 3e6, 4e6, 5e6])].product_type.value_counts()
train_df[train_df.product_type=="Investment"].price_doc.value_counts().head(20)
train_df[~(train_df.product_type=="Investment")].price_doc.value_counts().head(20)

print( "\nAmong", train_df[(train_df.product_type=="Investment")].price_doc.count(), 
      "investment sales, there were only", 
      train_df[(train_df.product_type=="Investment")].price_doc.nunique(), "unique prices.\n")
print( "Among", train_df[~(train_df.product_type=="Investment")].price_doc.count(), 
      "owner-occupant sales, there were", 
      train_df[~(train_df.product_type=="Investment")].price_doc.nunique(), "unique prices." )

#%% Price Year
print 'Yearly prices'

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
train_df["year"], train_df["month"], train_df["day"] = train_df["timestamp"].dt.year,train_df["timestamp"].dt.month,train_df["timestamp"].dt.day
train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: str(x)[:4]+str(x)[5:7])

train_df["count"] = 1
count_year = train_df.groupby("year").count().reset_index()
sns.barplot(count_year["year"],count_year["count"])

train_df.groupby("yearmonth").aggregate(np.mean).reset_index()
plt.figure(figsize= (12,8))
plt.xticks(rotation="vertical")
sns.barplot(train_df["yearmonth"].values,train_df["price_doc"].values)

train_df.groupby("build_year").aggregate(np.mean).reset_index()
plt.figure(figsize= (20,12))
plt.xticks(rotation="vertical")
sns.barplot(train_df["build_year"],train_df["full_sq"])
#%% Living square
print 'Size of the houses'

train_df.loc[train_df.full_sq < train_df.life_sq, "full_sq"] = train_df.life_sq

yr_grp = train_df.groupby("year").mean().reset_index()
fig, ax = plt.subplots(ncols=2)
fig.set_size_inches(10, 3)

sns.barplot(data=yr_grp, x="year", y="full_sq", orient="v", ax=ax[0])
ax[0].set_title("full_sq over the years")

sns.barplot(data=yr_grp, x="year", y="life_sq", orient="v", ax=ax[1])
ax[1].set_title("life_sq over the years")

#%% Correlation
print 'Correlation'

def corr_plot(dataframe, top_n, target, fig_x, fig_y):
    corrmat = dataframe.corr()
    #top_n - top n correlations +1 since price is included
    top_n = top_n + 1 
    cols = corrmat.nlargest(top_n, target)[target].index
    cm = np.corrcoef(train_df[cols].values.T)
    f, ax = plt.subplots(figsize=(fig_x,fig_y))
    sns.set(font_scale=1.25)
    cmap = plt.cm.viridis
    hm = sns.heatmap(cm, cbar=False, annot=True, square=True,cmap = cmap, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return cols
corr_20 = corr_plot(train_df, 20, 'price_doc', 10,10)

sns.heatmap(train_df[["full_sq", "life_sq", "num_room", "price_doc"]].corr())
#%%
print 'Macro heatmap'
macro_columns = macro_df.columns
corr_df=pd.DataFrame(columns=['feature','pearson', 'kendall', 'spearman'])
corr=macro_df[macro_columns].corr(method='spearman')
fig, ax = plt.subplots(figsize=(50,50))         # Sample figsize in inches
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)

#%%
print 'Training heatmap'
train_columns = train_df.columns
corr_df=pd.DataFrame(columns=['feature','pearson', 'kendall', 'spearman'])
corr=train_df[train_columns].corr(method='spearman')
fig, ax = plt.subplots(figsize=(150,150))         # Sample figsize in inches
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)

print 'Housemarket heatmap'
house_market_columns = house_market_df.columns
corr_df=pd.DataFrame(columns=['feature','pearson', 'kendall', 'spearman'])
corr=house_market_df[house_market_columns].corr(method='spearman')
fig, ax = plt.subplots(figsize=(50,50))         # Sample figsize in inches
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)
#%% Missing data in Train data

print 'Importing'
train_df = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/train.csv", parse_dates=['timestamp'])
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

print 'Missing data'

missing = train_df.isnull().sum(0).reset_index()
missing.columns = ['column', 'count']
missing = missing.sort_values(by = 'count', ascending = False).loc[missing['count'] > 0]
missing['percentage'] = missing['count'] / float(train_df.shape[0]) * 100
ind = np.arange(missing.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,18))
rects = ax.barh(ind, missing.percentage.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing.column.values, rotation='horizontal')
ax.set_xlabel("Precentage of missing values %", fontsize = 14)
ax.set_title("Number of missing values in each column", fontsize = 18)
plt.show()

#%% Missing data in Macro data

print 'Importing'
macro_df = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/macro.csv", parse_dates=['timestamp'])
stype_df = macro_df.dtypes.reset_index()
stype_df.columns = ["Count", "Column Type"]
stype_df.groupby("Column Type").aggregate('count').reset_index()

print 'Missing data'

missing = macro_df.isnull().sum(0).reset_index()
missing.columns = ['column', 'count']
missing = missing.sort_values(by = 'count', ascending = False).loc[missing['count'] > 0]
missing['percentage'] = missing['count'] / float(train_df.shape[0]) * 100
ind = np.arange(missing.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,18))
rects = ax.barh(ind, missing.percentage.values, color='g')
ax.set_yticks(ind)
ax.set_yticklabels(missing.column.values, rotation='horizontal')
ax.set_xlabel("Precentage of missing values %", fontsize = 14)
ax.set_title("Number of missing values in each column", fontsize = 18)
plt.show()

#%% Xgboost Train

print 'XGBoost feature importance X'

for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.title('Feature importance area')
plt.show()

#%% XGBoost macro

print 'XGBoost execution'

for f in macro_df.columns:
    if macro_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(macro_df[f].values)) 
        macro_df[f] = lbl.transform(list(macro_df[f].values))
        
macro_y = macro_df.gdp_annual.values
macro_X = macro_df.drop(["timestamp", "cpi", "gdp_annual","gdp_quart_growth", "gdp_deflator", "oil_urals", "gdp_quart", "gdp_annual_growth"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 12,
    'subsample': 0.3,
    'colsample_bytree': 0.3,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dmacro = xgb.DMatrix(macro_X, macro_y, feature_names=macro_X.columns.values)
moodel = xgb.train(dict(xgb_params, silent=0), dmacro, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(moodel, max_num_features=50, height=0.8, ax=ax)
plt.show()
#%% Correlation gender and age
print 'Correlation gender and age'

## Correlation
corrmat = train_df.ix[:,41:67].corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=.8, square=True,xticklabels=True,yticklabels=True,cbar=False,annot=True)


print 'Top 20 correlated variables'

# Top 20 correlated variables
corrmat = train_df.corr()
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'price_doc')['price_doc'].index
cm = np.corrcoef(train_df[cols].values.T)
f, ax = plt.subplots(figsize=(12, 12))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


#%% Feature selection

print 'Feature selection'
ulimit = np.percentile(train_df.price_doc.values, 99.5)
llimit = np.percentile(train_df.price_doc.values, 0.5)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit

col = "full_sq"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=np.log1p(train_df.full_sq.values), y=np.log1p(train_df.price_doc.values), size=10)
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of Total area in square metre', fontsize=12)
plt.show()

#%% Floor information

print 'Floor information'

col = "life_sq"
train_df[col].fillna(0, inplace=True)
ulimit = np.percentile(train_df[col].values, 95)
llimit = np.percentile(train_df[col].values, 5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=np.log1p(train_df.life_sq.values), y=np.log1p(train_df.price_doc.values), 
              kind='kde', size=10)
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of living area in square metre', fontsize=12)
plt.show()

plt.figure(figsize=(12,8))
sns.countplot(x="floor", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

grouped_df = train_df.groupby('floor')['price_doc'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(grouped_df.floor.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

plt.figure(figsize=(12,8))
sns.countplot(x="max_floor", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Max floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(x="max_floor", y="price_doc", data=train_df)
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Max Floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


#%% Missing values

ktype_df = house_market_df.dtypes.reset_index()
ktype_df.columns = ["Count", "Column Type"]
ktype_df.groupby("Column Type").aggregate('count').reset_index()

print 'Missing data'

missing = house_market_df.isnull().sum(0).reset_index()
missing.columns = ['column', 'count']
missing = missing.sort_values(by = 'count', ascending = False).loc[missing['count'] > 0]
missing['percentage'] = missing['count'] / float(train_df.shape[0]) * 100
ind = np.arange(missing.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,18))
rects = ax.barh(ind, missing.percentage.values, color='b')
ax.set_yticks(ind)
ax.set_yticklabels(missing.column.values, rotation='horizontal')
ax.set_xlabel("Precentage of missing values %", fontsize = 14)
ax.set_title("Number of missing values in each column", fontsize = 18)
plt.show()

#%% XGBoost feature importance

print 'XGBoost execution'

for f in house_market_df.columns:
    if house_market_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(house_market_df[f].values)) 
        house_market_df[f] = lbl.transform(list(house_market_df[f].values))
        
house_market_y = house_market_df.price_doc.values
house_market_X = house_market_df.drop(["id", "timestamp", "full_sq"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dhouse_market = xgb.DMatrix(house_market_X, house_market_y, feature_names=house_market_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dhouse_market, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.title('Feature importance of size house combined dataset')
plt.show()

#%% Interesting combinations

print "Normalizing interesting count combinations"

PC = train_df.price_doc.value_counts()
PU = train_df.price_doc.count()

print "Build year prizes"
train_df = train_df[train_df.build_year != 0.0]
train_df = train_df[train_df.build_year != 1.0]
train_df = train_df[train_df.build_year != 3.0]
train_df = train_df[train_df.build_year != 20.0]
train_df = train_df[train_df.build_year != 71.0]
train_df = train_df[train_df.build_year != 215.0]
train_df = train_df[train_df.build_year != 4965.0]
train_df = train_df[train_df.build_year != 20052009.0]
BYP = train_df.groupby(['build_year', 'price_doc']).size()
BYP = BYP.unstack()
BYP = BYP.fillna(0)
BYP_Norm = BYP.div(BYP.sum(axis=1), axis=0)

print "Location prizes"
LP = train_df.groupby(['sub_area', 'price_doc']).size()
LP = LP.unstack()
LP = LP.fillna(0)
LP_Norm = LP.div(LP.sum(axis=1), axis=0)
LP_stand = (LP - LP.mean()) / (LP.max() - LP.min())

print 'Build Year - Area'
BYA = train_df.groupby(['sub_area', 'build_year']).size()
BYA = BYA.unstack()
BYA = BYA.fillna(0)
BYA_Norm = BYA.div(BYA.sum(axis=1), axis=0)
BYA_stand = (BYA - BYA.mean()) / (BYA.max() - BYA.min())

print 'Product Type - Area'
PTP = train_df.groupby(['sub_area', 'product_type']).size()
PTP = PTP.unstack()
PTP = PTP.fillna(0)
PTP_Norm = PTP.div(PTP.sum(axis=1), axis=0)
PTP_stand = (PTP - PTP.mean()) / (PTP.max() - PTP.min())

print 'Product Type - Prices'
PCP = train_df.groupby(['price_doc', 'product_type']).size()
PCP = PCP.unstack()
PCP = PCP.fillna(0)
PCP_Norm = PCP.div(PCP.sum(axis=1), axis=0)
PCP_stand = (PCP - PCP.mean()) / (PCP.max() - PCP.min())


print 'Ecology - Area'
EA = train_df.groupby(['sub_area', 'ecology']).size()
EA = EA.unstack()
EA = EA.fillna(0)
EA_Norm = EA.div(EA.sum(axis=1), axis=0)
EA_stand = (EA - EA.mean()) / (EA.max() - EA.min())

print 'Ecology - Prices'
EP = train_df.groupby(['price_doc', 'ecology']).size()
EP = EP.unstack()
EP = EP.fillna(0)
EP_Norm = EP.div(EP.sum(axis=1), axis=0)
EP_stand = (EP - EP.mean()) / (EP.max() - EP.min())

print 'Number of rooms - Prices'
NRP = train_df.groupby(['num_room', 'price_doc']).size()
NRP = NRP.unstack()
NRP = NRP.fillna(0)
NRP_Norm = NRP.div(NRP.sum(axis=1), axis=0)
NRP_stand = (NRP - NRP.mean) / (NRP.max() - NRP.min())

print 'Build year - Timestamp'
TBY = train_df.groupby(['timestamp', 'build_year']).size()
TBY = TBY.unstack()
TBY = TBY.fillna(0)
TBY_Norm = TBY.div(TBY.sum(axis=1), axis=0)
TBY_stand = (TBY - TBY.mean()) / (TBY.max() - TBY.min())

print 'Build year - Build Materia'
RBC = train_df.groupby(['raion_build_count_with_material_info', 'build_year']).size()
RBC = RBC.unstack()
RBC = RBC.fillna(0)
RBC_Norm = RBC.div(RBC.sum(axis=1), axis=0)
RBC_stand = (RBC - RBC.mean()) / (RBC.max() - RBC.min())

print 'Appartment in area'
SAAB = house_market_df.groupby(['sub_area', 'apartment_build']).size()
SAAB = SAAB.unstack()
SAAB = SAAB.fillna(0)
SAAB_Norm = SAAB.div(SAAB.sum(axis=1), axis=0)
SAAB_stand = (SAAB - SAAB.mean()) / (SAAB.max() - SAAB.min())
