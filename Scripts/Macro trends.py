# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:11:24 2017

@author: Gebruiker
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy.polynomial.chebyshev import *
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

pd.options.mode.chained_assignment = None  
pd.set_option('display.max_columns', 500)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input director
#%% Importing Train
read_columns= ['timestamp', 'oil_urals', 'gdp_quart_growth', 'cpi', 'usdrub', \
                'salary_growth', 'unemployment', 'average_provision_of_build_contract_moscow', 'mortgage_rate', \
                 'deposits_rate','deposits_growth','rent_price_3room_eco',\
                 'rent_price_3room_bus']

print 'importing'
train_df = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/train.csv", usecols=['timestamp','price_doc','full_sq'])
train_df.shape
print 'head Train database'
train_df.head()

print 'importing Macro'
macro_df = pd.read_csv("D:/School/School/Master/Jaar_1/Machine Learning in Practice/2nd Competition/Data/macro.csv",usecols=read_columns)
macro_df.shape
print 'head Macro database'
macro_df.head()

#%%
def condition_train(value, col):
    vals = (macro_df[macro_df['mo_ye'] == value])
    
    ret = vals[col].asobject
  
    ret = ret[0]

    return ret

def condition_test(value, col):
    vals = (macro[macro['mo_ye'] == value])

    ret = vals[col].asobject

    ret = ret[0]

    return ret

def condition(value,col):
    vals = (macro_df[macro_df['timestamp'] == value])
    ret=vals[col].asobject
    ret=ret[0]

    return ret

def init_anlz_file():

    anlz_df = train_df
    for clmn in read_columns:
        if clmn == 'timestamp':
            continue
        anlz_df[clmn] = np.nan
        anlz_df[clmn] = anlz_df['timestamp'].apply(condition, col=clmn)
        print(clmn)
    return anlz_df

### Read Data for macro analysis
anlz_df=init_anlz_file()

#%%
##------------------------ SERVICE ROUTINES ----------------------------------- ###
methods=['pearson', 'kendall', 'spearman']
def plot_grouped_trends(df,feat1,feat2,corr_df):
   
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    x=df.index.values
    ch=chebfit(x,df[feat1].values,7)
    trendf1=chebval(x,ch)
    ax[0].plot(x,df[feat1].values,x,trendf1)
    ax[0].set_ylabel(feat1)
    ax[0].set_title('Chart '+feat1+' vs trend' )
    ax[0].set_xlabel('months count')
    ch2=chebfit(x,df[feat2].values,7)
    trendf2=chebval(x,ch2)
    ax[1].plot(x,df[feat2].values,x,trendf2)
    ax[1].set_ylabel(feat2)
    ax[1].set_title('Chart '+feat2+' vs trend' )
    ax[1].set_xlabel('months count')
    ##### do here two charts density distribition
    
    ls=[feat2]
    for method in methods:
        corr=df[[feat1,feat2]].corr(method=method)
        ls.append(corr[feat1][1])
    corr_df.loc[len(corr_df)]=ls
#%%
anlz_df['timestamp']=pd.to_datetime(anlz_df['timestamp'])
anlz_df['mo_ye']=anlz_df['timestamp'].apply(lambda x: x.strftime('%m-%Y'))
anlz_df['price_per_sqm']=anlz_df['price_doc']/anlz_df['full_sq']


macro_columns = ['price_doc','price_per_sqm','full_sq','oil_urals', 'gdp_quart_growth', 'cpi', 'usdrub', \
                'salary_growth', 'unemployment', 'average_provision_of_build_contract_moscow', 'mortgage_rate', \
                 'deposits_rate','deposits_growth','rent_price_3room_eco',\
                 'rent_price_3room_bus']
macro_df=pd.DataFrame(anlz_df.groupby('mo_ye')[macro_columns].mean())
macro_df.reset_index(inplace=True)


macro_df['mo_ye']=pd.to_datetime(macro_df['mo_ye'])
macro_df=macro_df.sort_values(by='mo_ye')


macro_df.reset_index(inplace=True)
macro_df.drop(['index'],axis=1,inplace=True)

#%%
corr_df=pd.DataFrame(columns=['feature','pearson', 'kendall', 'spearman'])
corr=macro_df[macro_columns].corr(method='spearman')
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)

#%%
for feat in macro_columns:
    if (feat=='price_doc'):
        continue
    plot_grouped_trends(macro_df,'price_doc',feat,corr_df)
#%%
print 'Median Oil with House Prices'

groeped_df = macro_df.groupby('oil_urals')['price_doc'].aggregate(np.median).reset_index()
groeped_df = groeped_df.sort(['price_doc']).reset_index(drop=True)
plt.figure(figsize=(25,8))
sns.barplot(groeped_df.oil_urals.values, groeped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Oil prices', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

print 'Median Mortgage with House Prices'

groeped_df = macro_df.groupby('mortgage_rate')['price_doc'].aggregate(np.median).reset_index()
groeped_df = groeped_df.sort(['price_doc']).reset_index(drop=True)
plt.figure(figsize=(25,8))
sns.barplot(groeped_df.mortgage_rate.values, groeped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Mortgage rate', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

print 'Median Unemployment with House Prices'

groeped_df = macro_df.groupby('unemployment')['price_doc'].aggregate(np.median).reset_index()
groeped_df = groeped_df.sort(['price_doc']).reset_index(drop=True)
plt.figure(figsize=(25,8))
sns.barplot(groeped_df.unemployment.values, groeped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Unemployment', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
