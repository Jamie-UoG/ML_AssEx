#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest,  mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif

min_max_scaler = preprocessing.MinMaxScaler()
# c_params = np.array([2])

# parameters = {'max_depth': c_params}


# Load training and testing data
xdata = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
ydata = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

xdata = min_max_scaler.fit_transform(xdata)
X_test = min_max_scaler.fit_transform(X_test)

# xdata = VarianceThreshold(0.01).fit_transform(xdata)
# print xdata.shape



mutinfo = mutual_info_classif(xdata,ydata, n_neighbors=5)
delarray=[]
for i in range((len(mutinfo)-1),-1,-1):
	if mutinfo[i]<0.01:
		delarray.append(i)
xdata = np.delete(xdata,delarray,axis=1)
print xdata.shape

xdata = SelectKBest(chi2, k = 80).fit_transform(xdata,ydata)
# print xdata.shape

#split data in half
#X_train,X_test=xdata[:len(xdata)/2],xdata[len(xdata)/2:]
#y_train,ytest=ydata[:len(ydata)/2],ydata[len(ydata)/2:]

#3/4 data for training
# x_train,x_test=xdata[:len(xdata)/4*3],xdata[len(xdata)/4*3:]
# y_train,y_test=ydata[:len(ydata)/4*3],ydata[len(ydata)/4*3:]

#1/4 data for training
x_train,x_test=xdata[len(xdata)/4*3:],xdata[:len(xdata)/4*3]
y_train,y_test=ydata[len(ydata)/4*3:],ydata[:len(ydata)/4*3]

#dont split the data
# x_train = xdata
# y_train = ydata

# rf = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters)
rf = RandomForestClassifier(n_estimators=100000)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
score = rf.score(x_test,y_test)
print rf.get_params()

# print importance
print "score ", score
print y_pred

