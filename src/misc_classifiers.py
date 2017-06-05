#!/usr/bin/env python2

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np


DATADIR = "../data/"

# check for cleaned data files
try:
    trainfn = "one-hot_median_filled_train.csv"
    pricefn = "train_prices.csv"
    testfn  = "one-hot_median_filled_test.csv"
    trainfile = pd.read_csv(DATADIR + trainfn)
    pricefile = pd.read_csv(DATADIR + pricefn)
    testfile  = pd.read_csv(DATADIR + testfn)
except IOError as e: 
    print "Could not read file: {}".format(e)

sc = StandardScaler()

clf = SGDClassifier(loss="hinge", penalty="l2")
tf = trainfile[["life_sq", 'build_year', 'num_room', 'raion_popul', 'shopping_centers_raion']]
sc.fit(tf)
tf = sc.transform(tf)

tt = sc.transform(testfile[["life_sq", 'build_year', 'num_room', 'raion_popul', 'shopping_centers_raion']])

clf.fit(tf, pricefile['price_doc'])

pred = clf.predict(tt)
testids = np.arange(30474, 38135)

print pred
print testids
fin = np.stack(tt[0], pred)

print fin

#,full_sq,life_sq,floor,max_floor,material,build_year,num_room,kitch_sq,state,area_m,raion_popul,green_zone_part,indust_part,children_preschool,preschool_quota,preschool_education_centers_raion,children_school,school_quota,school_education_centers_raion,school_education_centers_top_20_raion,hospital_beds_raion,healthcare_centers_raion,university_top_20_raion,sport_objects_raion,additional_education_raion,culture_objects_top_25_raion,shopping_centers_raion,office_raion,full_all,male_f,female_f,young_all,young_male,young_female,work_all,work_male,work_female,ekder_all,ekder_male,ekder_female,0_6_all,0_6_male,0_6_female,7_14_all,7_14_male,7_14_female,0_17_all,0_17_male,0_17_female,16_29_all,16_29_male,16_29_female,0_13_all,0_13_male,0_13_female,raion_build_count_with_material_info,build_count_block,build_count_wood,build_count_frame,build_count_brick,build_count_monolith,build_count_panel,build_count_foam,build_count_slag,build_count_mix,raion_build_count_with_builddate_info,build_count_before_1920,build_count_1921-1945,build_count_1946-1970,build_count_1971-1995,build_count_after_1995,ID_metro,metro_min_avto,metro_km_avto,metro_min_walk,metro_km_walk,kindergarten_km,school_km,park_km,green_zone_km,industrial_km,water_treatment_km,cemetery_km,incineration_km,railroad_station_walk_km,railroad_station_walk_min,ID_railroad_station_walk,railroad_station_avto_km,railroad_station_avto_min,ID_railroad_station_avto,public_transport_station_km,public_transport_station_min_walk,water_km,mkad_km,ttk_km,sadovoe_km,bulvar_ring_km,kremlin_km,big_road1_km,ID_big_road1,big_road2_km,ID_big_road2,railroad_km,zd_vokzaly_avto_km,ID_railroad_terminal,bus_terminal_avto_km,ID_bus_terminal,oil_chemistry_km,nuclear_reactor_km,radiation_km,power_transmission_line_km,thermal_power_plant_km

# vim: set et:ts=4:sw=4:
