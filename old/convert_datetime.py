import numpy as np
import pandas as pd
import datetime as dt
import os

CSV_PATH_PREFIX = 'MLiP/'
CSV_SRC_PREFIX = '../unzipped/'
events = pd.read_csv(CSV_SRC_PREFIX + 'events.csv', parse_dates=['timestamp'], infer_datetime_format=True)

first_5k_splitted = events.assign(
    date_year = lambda d: d['timestamp'].dt.year,
    date_month = lambda d: d['timestamp'].dt.month,
    date_day = lambda d: d['timestamp'].dt.day,
    date_hour = lambda d: d['timestamp'].dt.hour,
    date_minute = lambda d: d['timestamp'].dt.minute,
    date_second = lambda d: d['timestamp'].dt.second,
    date_weekday = lambda d: d['timestamp'].dt.weekday
    )

sorted_columns = ['event_id',
    'device_id',
    'timestamp',
    'longitude',
    'latitude',
    'date_year',
    'date_month',
    'date_day',
    'date_hour',
    'date_minute',
    'date_second',
    'date_weekday' ]

first_5k = first_5k_splitted.reindex(columns=sorted_columns)
first_5k.drop(['timestamp'], axis=1, inplace=True)

if not os.path.isfile(CSV_PATH_PREFIX + 'event_splitted_dt.csv'):
    first_5k.to_csv(CSV_PATH_PREFIX + 'event_splitted_dt.csv', index=False)
