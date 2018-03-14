#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif

min_max_scaler = preprocessing.MinMaxScaler()

# Load training and testing data
xdata = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
ydata = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

xdata = min_max_scaler.fit_transform(xdata)
X_test = min_max_scaler.fit_transform(X_test)

# dont split the data
x_train = xdata
y_train = ydata

# Split training data into 2 based on classification
epidata = []
stromadata = []
for i in range(len(y_train)):
	if y_train[i]==1:
		epidata.append(i)
	elif y_train[i]==2:
		stromadata.append(i)

# #Calculate priors for each class
epiprior=1.0*len(epidata)/len(x_train)
stromaprior=1.0*len(stromadata)/len(x_train)

#Find mean and standard deviation for all x in epidata
epimeans = []
epistddevs = []
for i in range(x_train.shape[1]):	#for all features
	meansum = 0.0
	temp = []
	for j in range(len(epidata)):	#for all training data classified as epi
		meansum += x_train[epidata[j]][i]	#add up coulmn of feature[i]
		temp.append(x_train[epidata[j]][i])
	mean = meansum / len(epidata)
	stddev = np.std(temp)
	epimeans.append(mean)
	epistddevs.append(stddev)

# #Find mean and standard deviation for all x stromadata
stromameans = []
stromastddevs = []
for i in range(x_train.shape[1]):	#for all features
	meansum = 0.0
	temp = []
	for j in range(len(stromadata)):	#for all training data classified as epi
		meansum += x_train[stromadata[j]][i]	#add up column of feature[i]
		temp.append(x_train[stromadata[j]][i])	#represent column of data as a vector
	mean = 1.0*meansum / len(stromadata)
	stddev = np.std(temp)
	stromameans.append(mean)
	stromastddevs.append(stddev)

#For each row of test data, find probabilty of epi and prob. of stroma
onecount = 0
twocount=0
y_pred = []
for i in range(len(X_test)):#for each row of test data
	#epiprob
	epiprob=epiprior
	temp=[]
		#find normal dist for all columns, calc product of all normal dists, mult by epiprior
	for j in range(X_test.shape[1]):#for all features
		epiprob *= 1.0/np.sqrt(2.0*np.pi*epistddevs[j])
		epiprob *= np.exp((-0.5/epistddevs[j])*(X_test[i][j]-epimeans[j])**2)
	#stromaprob
	stromaprob = stromaprior
	temp=[]
	for k in range(X_test.shape[1]):#for all features
		stromaprob *= 1.0/np.sqrt(2.0*np.pi*stromastddevs[k])
		stromaprob *= np.exp((-0.5/stromastddevs[k])*(X_test[i][k]-stromameans[k])**2)

	stromaprob = stromaprob/(stromaprob+epiprob)
	if epiprob>stromaprob:
		y_pred.append(1)
	elif stromaprob>epiprob:
		y_pred.append(2)

print (y_pred)

# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
# for i in range(n_points):
# 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")