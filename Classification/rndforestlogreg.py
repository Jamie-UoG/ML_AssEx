#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif

min_max_scaler = preprocessing.MinMaxScaler()
rc_params = np.array([1,2,3,4,5,6,7,8,9,10])
rparameters = {}

lc_params = np.array([7.0])
lparameters = {'C': lc_params, 'penalty': ["l1","l2"]}

# Load training and testing data
xdata = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
ydata = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

#minimise the data between 0 and 1
xdata = min_max_scaler.fit_transform(xdata)
X_test = min_max_scaler.fit_transform(X_test)


'''
Random forest
'''
selector = SelectKBest(f_classif, k = 60)
rdata = selector.fit_transform(xdata,ydata)
rX_test = selector.transform(X_test)

#split data in half
#X_train,X_test=rdata[:len(rdata)/2],rdata[len(rdata)/2:]
#y_train,ytest=ydata[:len(ydata)/2],ydata[len(ydata)/2:]

#3/4 data for training
# x_train,x_test=rdata[:len(rdata)/4*3],rdata[len(rdata)/4*3:]
# y_train,y_test=ydata[:len(ydata)/4*3],ydata[len(ydata)/4*3:]

#1/4 data for training
# x_train,x_test=rdata[len(rdata)/4*3:],rdata[:len(rdata)/4*3]
# y_train,y_test=ydata[len(ydata)/4*3:],ydata[:len(ydata)/4*3]

#dont split the data
x_train = rdata
y_train = ydata


rf = GridSearchCV(RandomForestClassifier(n_estimators=10000, n_jobs=-1), rparameters)
rf.fit(x_train,y_train)
r_pred = rf.predict_proba(rX_test)
# rscore = rf.score(x_test,y_test)


'''
Logistic regression
'''
selector = SelectKBest(chi2, k = 32)
ldata = selector.fit_transform(xdata,ydata)
lX_test = selector.transform(X_test)

#split data in half
#X_train,X_test=ldata[:len(ldata)/2],ldata[len(ldata)/2:]
#y_train,ytest=ydata[:len(ydata)/2],ydata[len(ydata)/2:]

#3/4 data for training
# x_train,x_test=ldata[:len(ldata)/4*3],ldata[len(ldata)/4*3:]
# y_train,y_test=ydata[:len(ydata)/4*3],ydata[len(ydata)/4*3:]

#1/4 data for training
x_train,x_test=ldata[len(ldata)/4*3:],ldata[:len(ldata)/4*3]
y_train,y_test=ydata[len(ydata)/4*3:],ydata[:len(ydata)/4*3]

#dont split the data
# x_train = ldata
# y_train = ydata

logreg = GridSearchCV(LogisticRegression(class_weight='balanced'), lparameters)
logreg.fit(x_train,y_train)
l_pred = logreg.predict_proba(lX_test)
# lscore = logreg.score(x_test,y_test)


'''
Ensemble of random forest & logistic regression
'''
ensemble = []
for i in range(r_pred.shape[0]):
	epiprob = r_pred[i][0] + 1.1*(l_pred[i][0])
	stromaprob = r_pred[i][1] + 1.1*(l_pred[i][1])
	if epiprob > stromaprob:
		ensemble.append(1)
	elif stromaprob > epiprob:
		ensemble.append(2)
	elif stromaprob == epiprob:
		print "EQUAL"
		ensemble.append(rand.randint(1,2))	 

# correct = 0
# incorr = 0
# for i in range(len(x_test)):
# 	if ensemble[i]==y_test[i]:
# 		correct += 1
# 	else:
# 		incorr += 1


# print "rnd forest score: ", rscore
# print r_pred
# print "log regression score: ", lscore
# print l_pred
# print "ensemble score: ", (1.0*correct / (correct + incorr))

y_pred=ensemble

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
# for i in range(n_points):
# 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")
