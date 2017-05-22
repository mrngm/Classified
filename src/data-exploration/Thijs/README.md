# Data exploration


## Feature importance

Feature importance code, as shown [here](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-sberbank) (with minor edits)

<details>
<summary>Code</summary>

```
#%% XGBoost Variable Importance

train_df = trainset

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

#%% Feature Importance list and plot

sorted_scores = sorted(model.get_score().items(), key=operator.itemgetter(1), reverse=True)

for i in range(0,20):
    print (str(i+1) + ". " + str(sorted_scores[i][0]) + ": " + str(sorted_scores[i][1]))

# plot the important features #
fig, ax = plt.subplots()
xgb.plot_importance(model, max_num_features=20, height=0.8, ax=ax)
plt.show()
```

</details>

This returns a list and a plot (plot shown below).

![alt text](images/feature_importance.png "Feature importance")

I'll take a closer look at the features that have the highest importance in this list.

### full_sq

_full_sq: total area in square meters, including loggias, balconies and other non-residential areas._

<details>
<summary>Code and filtering</summary>


```
plt.figure()
plt.plot(trainset['full_sq'], trainset['price_doc'], "o",
         color="g", ms=5)
plt.show()
```

![alt text](images/full_sq_all.png "full_sq with outlier")

Filter out the outlier and plot new dataset:

```
# Remove outlier
filtered = trainset[trainset['full_sq'] < 4000]

sns.jointplot(x="full_sq", y="price_doc", data=filtered,
              color="g", size=8, s=10)
```

</details>

![alt text](images/full_sq_filtered.png "full_sq without outlier")

Most houses are around 50 square meters, and go up to 100, with some houses going up to 200 or more.

### life_sq

_life_sq: living area in square meters, excluding loggias, balconies and other non-residential areas_

<details>
<summary>Code and filtering</summary>


```
plt.figure()
plt.plot(trainset['life_sq'], trainset['price_doc'], "o",
         color="g", ms=5)
plt.show()
```

![alt text](images/life_sq_all.png "life_sq with outlier")

Filter out the outlier and plot new dataset:

```
# Remove outlier
filtered = trainset[trainset['life_sq'] < 7000]

sns.jointplot(x="life_sq", y="price_doc", data=filtered,
              color="g", size=8, s=10)
```

</details>

![alt text](images/life_sq_filtered.png "life_sq without outlier")

Same as with full_sq, life_sq has most points centered around 50-100 square meters.

### floor

_floor: for apartments, floor of the building_

```
plt.figure(figsize=(12,8))
sns.countplot(x="floor", data=trainset)
plt.xticks(rotation='vertical')
```

![alt text](images/floor.png "floor")

Most apartments are between 0-10 floors high, something you would expect. Let's look at the median price per floor:

```
floor_medianprice = trainset.groupby('floor')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(x="floor", y="price_doc", data=floor_medianprice)
plt.xticks(rotation='vertical')
plt.ylabel('Median price')
```

![alt text](images/floor_median.png "median price per floor")

It's a bit hard to see, but there is a gradual growth in price. You can see small increases at 0 and 18. The data after 25 is probably distorted by few very high priced apartments.

### max_floor

_max_floor: number of floors in the building_

```
plt.figure(figsize=(12,8))
sns.countplot(x="max_floor", data=trainset)
plt.xticks(rotation='vertical')
```

![alt text](images/max_floor.png "max_floor")

The most frequent amount of floors are all over the place, so we can't really draw any conclusions from this. Let's again look at median prices:

```
max_floor_medianprice = trainset.groupby('max_floor')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(x="max_floor", y="price_doc", data=max_floor_medianprice)
plt.xticks(rotation='vertical')
plt.ylabel('Median price')
```

![alt text](images/max_floor_median.png "median price per max_floor")

There seems to be some growth in price when you have higher buildings, but this isn't certain.

### build_year

These are the unique values found in build_year (as integers):

```
[          0           1           3          20          71         215
        1691        1860        1876        1886        1890        1895
        1896        1900        1904        1905        1906        1907
        1910        1911        1912        1914        1915        1917
        1920        1924        1925        1926        1927        1928
        1929        1930        1931        1932        1933        1934
        1935        1936        1937        1938        1939        1940
        1941        1943        1946        1947        1948        1949
        1950        1951        1952        1953        1954        1955
        1956        1957        1958        1959        1960        1961
        1962        1963        1964        1965        1966        1967
        1968        1969        1970        1971        1972        1973
        1974        1975        1976        1977        1978        1979
        1980        1981        1982        1983        1984        1985
        1986        1987        1988        1989        1990        1991
        1992        1993        1994        1995        1996        1997
        1998        1999        2000        2001        2002        2003
        2004        2005        2006        2007        2008        2009
        2010        2011        2012        2013        2014        2015
        2016        2017        2018        4965    20052009 -2147483648]
```

There are some weird values here, like all years before 1000 (Moscow wasn't founded until around the 14th century), all values after 2018 (2018 seems reasonable as a future building project), and NaN. Filter and plot the remaining values:

```
# Remove outliers
filtered = trainset[~pd.isnull(trainset)]
filtered = filtered[(filtered.build_year > 1600)
    & (filtered.build_year != 4965)
    & (filtered.build_year != 20052009)]

sns.jointplot(x="build_year", y="price_doc", data=filtered,
              color="g", size=8, s=10)
```

![alt text](images/build_year.png "build_year without outliers")

Most buildings in the set are built after 1950, and more expensive buildings appear more frequently after 2000.

### state

_state: apartment condition_

I'm not sure what kind of scores they use for this, so first I printed the unique results, which resulted in this:

```
[  1.   2.   3.   4.  33.  nan]
```

It seems to be some sort of ranking from 1-4, but I don't know why the 33 is there (probably an outlier, as you can see in the next graph).

![alt text](images/state.png "state")

Most houses/apartments are in 1-3, with a lot less in 4 and almost no in 33 (which is probably a mistake anyway).

![alt text](images/build_year.png "median price per state")

Here you can see the median price per state. The price actually gets higher when the rating gets higher, so combined with the least amount of houses in state 4, it is probably the highest score, and 1 is the lowest.

### kitch_sq

_kitch_sq: kitchen area_

<details>
<summary>Code and filtering</summary>


```
plt.figure()
plt.plot(trainset['kitch_sq'], trainset['price_doc'], "o",
         color="g", ms=5)
plt.show()
```

![alt text](images/kitch_sq_all.png "kitch_sq, full data")

Get a close-up of the rest of the data points:

```
# Get close-up of data
filtered = trainset[trainset['kitch_sq'] < 500]

sns.jointplot(x="kitch_sq", y="price_doc", data=filtered,
              color="g", size=8, s=10)
```

</details>

![alt text](images/kitch_sq_filtered.png "kitch_sq without outlier")

Most kitchens in the set aren't larger than 20 square meters.

### num_room

_num_room: number of living rooms_

```
plt.figure()
sns.countplot(x="num_room", data=trainset)
```

![alt text](images/num_room.png "num_room")

2 and 1 living rooms are most frequent, followed by 3. All other values are much lower. Let's take a look at the median price per num_room:

```
num_room_medianprice = trainset.groupby('num_room')['price_doc'].aggregate(np.median).reset_index()

plt.figure()
sns.barplot(x="num_room", y="price_doc", data=num_room_medianprice)
plt.ylabel('Median price')
```

![alt text](images/num_room_median.png "median price per num_room")

As you can see, the median price rises further and further until the highest amount at 9, after which it falls.

## Timestamps

Let's take a look at the timestamps.

```
#%% Preparing data

trainset['year'] = trainset['timestamp'].apply(lambda x: x[:4])
trainset['year-month'] = trainset['timestamp'].apply(lambda x: x[:4] + "-" + x[5:7])

#%% Counting the amount of entries by timestamp
plt.figure()
sns.countplot(x="year", data=trainset)

plt.figure(figsize=(12,8))
sns.countplot(x="year-month", data=trainset)
plt.xticks(rotation='vertical')
```

This creates two images:

![alt text](images/timestamp-year.png "timestamp year")

![alt text](images/timestamp-yearmonth.png "timestamp year-month")

Nothing very remarkable about these images: a lot of the data comes from 2014, but this doesn't really tell us much. Let's take a look at how it influences the (median) price:

```
#%% Timestamps with median price

medianprice = trainset.groupby('year-month')['price_doc'].aggregate(np.median).reset_index().sort_values("year-month")

plt.figure(figsize=(12,8))
sns.barplot(x="year-month", y="price_doc", data=medianprice)
plt.xticks(rotation='vertical')
plt.ylabel('Median price')
```

![alt text](images/timestamp-medianprice.png "median price per year-month")

You can see a slight increase over time.