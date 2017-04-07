# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 19:34:00 2017

@author: Gebruiker
"""
from __future__ import division
"""
Programming Preparation
"""

import numpy as np
import pandas as pd
import datetime as dt
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
import itertools
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
#%%

#Using different classification machine learning algorithms to check for inconsistencies and problems
Universal = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Noise Eliminated Universal Files/Universal.csv', index_col = 0, dtype={'gender': object, 'group': object, 'phone_brand': object, 'device_model': object})
del Universal['group']
Universal.columns
#%%
unique_devices = np.unique(Universal['device_id'])
for u in unique_devices:
    device_data=Universal[Universal['device_id']==u]
        
    data_csv = open('./device_data_files/'+unicode(u)+'.txt', 'w')
    device_data.to_csv(data_csv, sep= ' ', header=False, index=False, columns=['phone_brand', 'device_model', 'app_id','category'], encoding = 'utf-8')
    data_csv.close()
    
    data_ID = open('./device_data_files/IDs/'+unicode(u)+'_IDs.txt', 'w')
    device_data.to_csv(data_ID, sep= ' ', header=False, index=False, columns=['app_id'], encoding = 'utf-8')
    data_ID.close()

    cat_ID = open('./device_data_files/Cat/'+unicode(u)+'_Cat.txt', 'w')
    device_data.to_csv(cat_ID, sep= ' ', header=False, index=False, columns=['category'], encoding = 'utf-8')
    cat_ID.close()

#%%



H = Universal.head(10)
#Stastics
sns.distplot(Universal.age.dropna(), color='#63EA55')
width = 1
axes = plt.gca()
axes.set_ylim([0.00,0.10])
axes.set_xlim([15,90])
plt.xlabel('Age')
sns.despine()
plt.show

F23 = (np.count_nonzero(Universal.group == 'F23-' )/4055391)*100
F24_26 = (np.count_nonzero(Universal.group == 'F24-26')/4055391)*100
F27_28 = (np.count_nonzero(Universal.group == 'F27-28')/4055391)*100
F29_32 = (np.count_nonzero(Universal.group == 'F29-32')/4055391)*100
F33_42 = (np.count_nonzero(Universal.group == 'F33-42')/4055391)*100
F43 = (np.count_nonzero(Universal.group == 'F43+')/4055391)*100
M22 = (np.count_nonzero(Universal.group == 'M22-' )/4055391)*100
M23_26 = (np.count_nonzero(Universal.group == 'M23-26')/4055391)*100
M27_28 = (np.count_nonzero(Universal.group == 'M27-28')/4055391)*100
M29_31 = (np.count_nonzero(Universal.group == 'M29-31')/4055391)*100
M32_38 = (np.count_nonzero(Universal.group == 'M32-38')/4055391)*100
M39 = (np.count_nonzero(Universal.group == 'M39+')/4055391)*100

Phone_brand = Universal.groupby('phone_brand').count()
Event = Universal.groupby('event_id').count()
Device = Universal.groupby('device_model').count()
Category = Universal.groupby('category').count()
Gender = Universal.groupby('gender').count()
Age = Universal.groupby('age').count()
Device_Info = Universal.groupby(['phone_brand','device_model']).count()
Location = Universal.groupby(['device_id','longitude', 'latitude']).count()
App_Activity = Universal.groupby(['category', 'is_installed','is_active']).count()

Universal = Universal.groupby('event_id').filter(lambda x: len(x) > 1)
Universal = Universal.groupby('device_id').filter(lambda x: len(x) > 1)
Universal = Universal.groupby('app_id').filter(lambda x: len(x) > 1)
Universal = Universal.groupby('label_id').filter(lambda x: len(x) > 1)

#Phone Brand all under the top 20 is going to be removed
#Event everything below 2 is going to be removed
#Category everything below 500 is going to be removed
#Device model everything below 100 is going to be removed
#Age everything below 10.000 is going to be removed

Event_limit = Event >= 2
Event_limit = Event_limit[Event_limit.device_id != True]
print Event_limit

Category_limit = Category >= 50
Category_limit = Category_limit[Category_limit.device_id != True]
print Category_limit

Phone_brand = Phone_brand.device_id.sort_values(ascending=1)
Event = Event.device_id.sort_values(ascending=1)
Device = Device.device_id.sort_values(ascending=1)
Category = Category.device_id.sort_values(ascending=1)
Gender = Gender.device_id.sort_values(ascending=1)
Age = Age.device_id.sort_values(ascending=1)
Location = Location.event_id.sort_values(ascending=1)
App_Activity = App_Activity.event_id.sort_values(ascending=1)
Device_Info = Device_Info.event_id.sort_values(ascending=1)

print Phone_brand
print Device_Info
print Event
print Device
print Category
print Gender
print Age
print Location
print App_Activity


#Average amount based on all labels present
Phone_brand_amount = (4055391/37)*0.01
print Phone_brand_amount
Category_amount = (4055391/429)*0.1
print Category_amount
Event_amount = (4055391/24897)*0.2
print Event_amount

#Dropping less popular values
#Based on phone_brand
Universal = Universal[Universal.phone_brand != 'VOXTEL']
Universal = Universal[Universal.phone_brand != 'DAKELE']
Universal = Universal[Universal.phone_brand != 'PHICOMM']
Universal = Universal[Universal.phone_brand != 'GAMMA']
Universal = Universal[Universal.phone_brand != 'CHINESE']
Universal = Universal[Universal.phone_brand != 'T-SERIES']
Universal = Universal[Universal.phone_brand != 'LAVA']
Universal = Universal[Universal.phone_brand != 'CHINA-QUADBAND-TV']
Universal = Universal[Universal.phone_brand != 'CUBE-TALK']
Universal = Universal[Universal.phone_brand != 'ZTE']
Universal = Universal[Universal.phone_brand != 'SAMSUNG-JITTERBUG']
Universal = Universal[Universal.phone_brand != 'HKS']
Universal = Universal[Universal.phone_brand != 'NEWMAN']
Universal = Universal[Universal.phone_brand != 'NIBURU']
Universal = Universal[Universal.phone_brand != 'LI-FENG']
Universal = Universal[Universal.phone_brand != 'SAMSUNG-GALAXY']
Universal = Universal[Universal.phone_brand != 'HISENSE']
Universal = Universal[Universal.phone_brand != 'ZUK']

#Based on category
Universal = Universal[Universal.category != 'violence comic']
Universal = Universal[Universal.category != 'tribe']
Universal = Universal[Universal.category != 'tennis']
Universal = Universal[Universal.category != 'shows']
Universal = Universal[Universal.category != 'safety Insurance']
Universal = Universal[Universal.category != 'reality show']
Universal = Universal[Universal.category != 'raising up game']
Universal = Universal[Universal.category != 'puzzel']
Universal = Universal[Universal.category != 'phone']
Universal = Universal[Universal.category != 'pet raising up']
Universal = Universal[Universal.category != 'other ball game']
Universal = Universal[Universal.category != 'noble']
Universal = Universal[Universal.category != 'love raising up']
Universal = Universal[Universal.category != 'knight game']
Universal = Universal[Universal.category != 'game-gem']
Universal = Universal[Universal.category != 'game-aircraft']
Universal = Universal[Universal.category != 'game-Zuma']
Universal = Universal[Universal.category != 'game-Rowing']
Universal = Universal[Universal.category != 'game-Motorcycle']
Universal = Universal[Universal.category != 'game-Finding fault']
Universal = Universal[Universal.category != 'football']
Universal = Universal[Universal.category != 'fighting game']
Universal = Universal[Universal.category != 'entertainment']
Universal = Universal[Universal.category != 'email']
Universal = Universal[Universal.category != 'cosplay']
Universal = Universal[Universal.category != 'airport']
Universal = Universal[Universal.category != 'World of Warcraft']
Universal = Universal[Universal.category != 'Western Mythology']
Universal = Universal[Universal.category != 'Vermicelli']
Universal = Universal[Universal.category != 'Puzzle']
Universal = Universal[Universal.category != 'Puzzles']
Universal = Universal[Universal.category != 'Other shares']
Universal = Universal[Universal.category != 'Music Games']
Universal = Universal[Universal.category != 'Mother']
Universal = Universal[Universal.category != 'Man playing favorites']
Universal = Universal[Universal.category != 'Make-up application']
Universal = Universal[Universal.category != 'Literacy Games']
Universal = Universal[Universal.category != 'KTV']
Universal = Universal[Universal.category != 'Information']
Universal = Universal[Universal.category != 'Hong Kong, Macao and Taiwan (aviation)']
Universal = Universal[Universal.category != 'H shares']
Universal = Universal[Universal.category != 'High-end hotel']
Universal = Universal[Universal.category != 'Guard tower defense game']
Universal = Universal[Universal.category != 'Games']
Universal = Universal[Universal.category != 'Financial Future']
Universal = Universal[Universal.category != 'Financial Futures']
Universal = Universal[Universal.category != 'Europa, the United States and Macao (aviation)']
Universal = Universal[Universal.category != 'Europa, the United States and Macao (Travel)']
Universal = Universal[Universal.category != 'Educational games']
Universal = Universal[Universal.category != 'Doctors']
Universal = Universal[Universal.category != 'Desktop Enhancements']
Universal = Universal[Universal.category != 'Clock']
Universal = Universal[Universal.category != 'Class animation community']
Universal = Universal[Universal.category != 'Chinese painting']
Universal = Universal[Universal.category != 'Chinese Classical Mythology']
Universal = Universal[Universal.category != 'Antique collection']
Universal = Universal[Universal.category != 'Adventure Game']
Universal = Universal[Universal.category != 'Academic Information']
Universal = Universal[Universal.category != '90s Japanese comic']
Universal = Universal[Universal.category != '80s Japanese comic']
Universal = Universal[Universal.category != 'Europe, the United States and Macao (aviation)']
Universal = Universal[Universal.category != 'Europe, the United States and Macao (Travel)']
Universal = Universal[Universal.category != 'Financial Futures']
Universal = Universal[Universal.category != 'Regional Aviation']
Universal = Universal[Universal.category != 'Simple']
Universal = Universal[Universal.category != 'Skin care applications']
Universal = Universal[Universal.category != 'Southeast Asia (aviation)']
Universal = Universal[Universal.category != 'Sports']
Universal = Universal[Universal.category != 'Sports Games']
Universal = Universal[Universal.category != 'Table Games']
Universal = Universal[Universal.category != 'Trust']
Universal = Universal[Universal.category != 'game-3D']
Universal = Universal[Universal.category != 'App Store']
Universal = Universal[Universal.category != 'Business Office']
Universal = Universal[Universal.category != 'Classical 1']
Universal = Universal[Universal.category != 'Housekeeping']
Universal = Universal[Universal.category != 'Journey to the West game']
Universal = Universal[Universal.category != 'MOBA']
Universal = Universal[Universal.category != 'Ninja']
Universal = Universal[Universal.category != 'Peace - Search']
Universal = Universal[Universal.category != 'Ping']
Universal = Universal[Universal.category != 'Tower Defense']
Universal = Universal[Universal.category != 'basketball']
Universal = Universal[Universal.category != 'dotal-lol']
Universal = Universal[Universal.category != 'farm']
Universal = Universal[Universal.category != 'game-tank']
Universal = Universal[Universal.category != 'household products']
Universal = Universal[Universal.category != 'math']
Universal = Universal[Universal.category != 'movie']
Universal = Universal[Universal.category != 'shopping sharing']
Universal = Universal[Universal.category != 'study abroad']
Universal = Universal[Universal.category != 'game-Business simulation']
Universal = Universal[Universal.category != 'Japan and South Korea (Travel)']
Universal = Universal[Universal.category != 'Xian Xia']
Universal = Universal[Universal.category != 'Hardware Related']
Universal = Universal[Universal.category != 'japanese comic and animation']
Universal = Universal[Universal.category != 'classical']
Universal = Universal[Universal.category != 'Beauty Nail']
Universal = Universal[Universal.category != 'Travel preferences']
Universal = Universal[Universal.category != 'Share Tour']
Universal = Universal[Universal.category != 'Engineering Drawing']
Universal = Universal[Universal.category != 'natural']
Universal = Universal[Universal.category != 'game-Horizontal version']
Universal = Universal[Universal.category != 'Turn based RPG game']
Universal = Universal[Universal.category != 'game-Bobble']
Universal = Universal[Universal.category != 'Southeast Asia (Travel)']
Universal = Universal[Universal.category != 'Jin Yong']
Universal = Universal[Universal.category != 'poker and chess']
Universal = Universal[Universal.category != 'Shushan']
Universal = Universal[Universal.category != 'Furniture']
Universal = Universal[Universal.category != 'Romance']
Universal = Universal[Universal.category != 'Senki']
Universal = Universal[Universal.category != 'Outlaws of the Marsh game']
Universal = Universal[Universal.category != 'Hong Kong, Macao and Taiwan (Travel)']


#Based on longitude
Universal = Universal[Universal.longitude != 1.0]
Universal = Universal[Universal.longitude != 0.0]

#Based on Device Model
Universal = Universal[Universal.device_model != 'Galaxy S7']
Universal = Universal[Universal.device_model != 'X']
Universal = Universal[Universal.device_model != 'Y11IT']
Universal = Universal[Universal.device_model != 'Galaxy Core Lite']
Universal = Universal[Universal.device_model != 'A30']
Universal = Universal[Universal.device_model != 'T708']
Universal = Universal[Universal.device_model != 'A850']
Universal = Universal[Universal.device_model != 'S2y']
Universal = Universal[Universal.device_model != 'Galaxy S2']
Universal = Universal[Universal.device_model != 'MI 4S']
Universal = Universal[Universal.device_model != 'Nexus 5']
Universal = Universal[Universal.device_model != 'P9 Plus']
Universal = Universal[Universal.device_model != 'Grand Memo 2']
Universal = Universal[Universal.device_model != 'Y1']
Universal = Universal[Universal.device_model != 'P9']
Universal = Universal[Universal.device_model != 'A850+']
Universal = Universal[Universal.device_model != 'Galaxy Tab 3 7.0']
Universal = Universal[Universal.device_model != 'A766']
Universal = Universal[Universal.device_model != '5316']
Universal = Universal[Universal.device_model != 'Galaxy Note']
Universal = Universal[Universal.device_model != '大神Note ']
Universal = Universal[Universal.device_model != '星星2号']
Universal = Universal[Universal.device_model != '小辣椒 7 ']
Universal = Universal[Universal.device_model != 'One max']

#Based on Age
Universal = Universal[Universal.age != 14]
Universal = Universal[Universal.age != 15]
Universal = Universal[Universal.age != 16]
Universal = Universal[Universal.age != 69]
Universal = Universal[Universal.age != 70]
Universal = Universal[Universal.age != 71]
Universal = Universal[Universal.age != 72]
Universal = Universal[Universal.age != 73]
Universal = Universal[Universal.age != 74]
Universal = Universal[Universal.age != 75]
Universal = Universal[Universal.age != 76]
Universal = Universal[Universal.age != 77]
Universal = Universal[Universal.age != 78]
Universal = Universal[Universal.age != 79]
Universal = Universal[Universal.age != 80]
Universal = Universal[Universal.age != 81]
Universal = Universal[Universal.age != 82]
Universal = Universal[Universal.age != 83]
Universal = Universal[Universal.age != 84]
Universal = Universal[Universal.age != 85]
Universal = Universal[Universal.age != 86]
Universal = Universal[Universal.age != 87]
Universal = Universal[Universal.age != 88]
Universal = Universal[Universal.age != 89]
Universal = Universal[Universal.age != 90]
Universal = Universal[Universal.age != 91]
Universal = Universal[Universal.age != 92]
Universal = Universal[Universal.age != 93]
Universal = Universal[Universal.age != 94]
Universal = Universal[Universal.age != 95]
Universal = Universal[Universal.age != 96]
Universal = Universal[Universal.age != 97]
Universal = Universal[Universal.age != 98]


#Classification
Universal = Universal.iloc[np.random.permutation(len(Universal))]
y = Universal['gender'].copy()
Universal.drop('gender', inplace=True, axis=1)
Universal.drop('group', inplace=True, axis=1)
Universal.drop('category', inplace=True, axis=1)
Universal.drop('timestamp', inplace=True, axis=1)
Universal.drop('phone_brand', inplace=True, axis=1)
Universal.drop('device_model', inplace=True, axis=1)
Universal.drop('app_id', inplace=True, axis=1)
Universal.drop('label_id', inplace=True, axis=1)
Universal.drop('event_id', inplace=True, axis=1)
Universal.drop('device_id', inplace=True, axis=1)
Universal = Universal.head(10000)
y = y.head(10000)
X_train, X_test, y_train, y_test = train_test_split(Universal, y, test_size=0.4, random_state=4)

Names = ['M', 'F']
class_names = Names
            
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# evaluate each model in turn
seed = 100
scoring = 'accuracy'

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=75, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
 
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
Names = ["M", "F"]
class_names = Names

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 5.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()
"""
#K-Nearest Neighbors
X_train, X_test, y_train, y_test = train_test_split(Universal, y, test_size=0.4, random_state=4)
k_range = list(range(1, 100))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

ax = plt.gca()
ax.grid(True)   
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

#Decision Trees
X_train, X_test, y_train, y_test = train_test_split(Universal, y, test_size=0.4, random_state=5)

models = []
models.append(('Bagging', BaggingClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('GBC', GradientBoostingClassifier()))

#evaluate each model in turn
seed = 100
scoring = 'accuracy'

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=75, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
 
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Ensemble Preparation
X_train, X_test, y_train, y_test = train_test_split(Universal, y, test_size=0.4, random_state=10)
#Ensemble Classification
kfold = model_selection.KFold(n_splits=100, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = KNeighborsClassifier()
estimators.append(('KNN', model2))
model3 = SVC()
estimators.append(('svm', model3))
model4 = LinearDiscriminantAnalysis()
estimators.append(('LDA', model4))
# create the ensemble model
ensemble = VotingClassifier(estimators)
Train_results_Classification = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
print(Train_results_Classification .mean())

#Ensemble Trees
kfold = model_selection.KFold(n_splits=100, random_state=seed)
# create the sub models
estimators = []
model1 = BaggingClassifier()
estimators.append(('Bagging', model1))
model2 = RandomForestClassifier()
estimators.append(('RFC', model2))
model3 = ExtraTreesClassifier()
estimators.append(('ETC', model3))
model4 = GradientBoostingClassifier()
estimators.append(('GBC', model4))
# create the ensemble model
ensemble = VotingClassifier(estimators)
Train_results_Boosting = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
print(Train_results_Boosting.mean())