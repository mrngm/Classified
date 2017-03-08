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


#Using different classification machine learning algorithms to check for inconsistencies and problems
Universal = pd.read_csv('D:/School/School/Master/Jaar_1/Machine Learning in Practice/Competition/Data/Noise Eliminated Universal Files/Universal_1.csv', index_col = 0, dtype={'gender': object, 'group': object, 'phone_brand': object, 'device_model': object})

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
Location = Universal.groupby(['device_id','longitude', 'latitude']).count()
App_Activity = Universal.groupby(['category', 'is_active']).count()

Phone_brand = Phone_brand.device_id.sort_values(ascending=1)
Event = Event.device_id.sort_values(ascending=1)
Device = Device.device_id.sort_values(ascending=1)
Category = Category.device_id.sort_values(ascending=1)
Gender = Gender.device_id.sort_values(ascending=1)
Age = Age.device_id.sort_values(ascending=1)
Location = Location.event_id.sort_values(ascending=1)
App_Activity = App_Activity.event_id.sort_values(ascending=1)

print Phone_brand
print Event
print Device
print Category
print Gender
print Age
print Location
print App_Activity





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