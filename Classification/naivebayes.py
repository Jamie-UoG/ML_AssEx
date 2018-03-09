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

print xdata.shape



xdata = min_max_scaler.fit_transform(xdata)
X_test = min_max_scaler.fit_transform(X_test)

# mutinfo = mutual_info_classif(xdata,ydata, n_neighbors=3)
# delarray=[]
# for i in range((len(mutinfo)-1),-1,-1):
# 	if mutinfo[i]<0.01:
# 		delarray.append(i)
# 		# print "DELET"
# print mutinfo		
# xdata = np.delete(xdata,delarray,axis=1)
# print xdata.shape
# xdata = VarianceThreshold(0.01).fit_transform(xdata)
# print xdata.shape

selector = SelectKBest(f_classif, k = 10)
xdata = selector.fit_transform(xdata,ydata)
X_test = selector.transform(X_test)



# print xdata.shape

#split data in half
#X_train,X_test=xdata[:len(xdata)/2],xdata[len(xdata)/2:]
#y_train,ytest=ydata[:len(ydata)/2],ydata[len(ydata)/2:]

#3/4 data for training
# x_train,x_test=xdata[:len(xdata)/4*3],xdata[len(xdata)/4*3:]
# y_train,y_test=ydata[:len(ydata)/4*3],ydata[len(ydata)/4*3:]

#1/4 data for training
# x_train,x_test=xdata[len(xdata)/4*3:],xdata[:len(xdata)/4*3]
# y_train,y_test=ydata[len(ydata)/4*3:],ydata[:len(ydata)/4*3]

# dont split the data
x_train = xdata
y_train = ydata

# # print x_train.shape[0]
# # print x_train.shape[1]
# # print y_train
# # print y_test

#Split training data into 2 based on classification
# epidata = []
# stromadata = []
# for i in range(len(y_train)):
# 	if y_train[i]==1:
# 		epidata.append(i)
# 	elif y_train[i]==2:
# 		stromadata.append(i)

# #Calculate priors for each class
# epiprior=1.0*len(epidata)/len(x_train)
# stromaprior=1.0*len(stromadata)/len(x_train)
# # print x_train
# # print 'EPI'
# # print epidata
# # print 'STROMA'
# # print stromadata
# # print len(epidata)
# # print x_train.shape[1]


#Find mean and standard deviation for all x in epidata
# epimeans = []
# epistddevs = []
# for i in range(x_train.shape[1]):	#for all features
# 	meansum = 0.0
# 	temp = []
# 	for j in range(len(epidata)):	#for all training data classified as epi
# 		meansum += x_train[epidata[j]][i]	#add up coulmn of feature[i]
# 		temp.append(x_train[epidata[j]][i])
# 	mean = meansum / len(epidata)
# 	stddev = np.std(temp)
# 	epimeans.append(mean)
# 	epistddevs.append(stddev)

# stddevs = []
# means=[]
# for i in range(xdata.shape[1]):	#for all features
# 	meansum = 0.0
# 	temp = []
# 	for j in range(len(xdata)):	#for all training data classified as epi
# 		meansum += xdata[j][i]	#add up coulmn of feature[i]
# 		temp.append(xdata[j][i])
# 	mean = meansum / len(xdata)
# 	stddev = np.std(temp)
# 	means.append(mean)
# 	stddevs.append(stddev)


# #Find mean and standard deviation for all x stromadata
# stromameans = []
# stromastddevs = []
# for i in range(x_train.shape[1]):	#for all features
# 	meansum = 0.0
# 	temp = []
# 	for j in range(len(stromadata)):	#for all training data classified as epi
# 		meansum += x_train[stromadata[j]][i]	#add up column of feature[i]
# 		temp.append(x_train[stromadata[j]][i])	#represent column of data as a vector
# 	mean = 1.0*meansum / len(stromadata)
# 	stddev = np.std(temp)
# 	stromameans.append(mean)
# 	stromastddevs.append(stddev)

#For each row of test data, find probabilty of epi and prob. of stroma
# print X_test.shape[1]
# onecount = 0
# twocount=0
# y_pred = []
# for i in range(len(X_test)):#for each row of test data
# 	#epiprob
# 	epiprob=0.0
# 	normprod = 1.0
# 	temp=[]
# 		#find normal dist for all columns, calc product of all normal dists, mult by epiprior
# 	for j in range(X_test.shape[1]):#for all features
# 		normdist = npr.normal(epimeans[j],epistddevs[j])
# 		temp.append(normdist)
# 	normprod=np.prod(temp)
# 	# print temp
# 	# print normdist

# 	epiprob = normprod * epiprior / 112
# 	# print normprod
# 	# print epiprior
# 	#stromaprob
# 	stromaprob=0.0
# 	normprod = 1.0
# 	temp=[]
# 	for k in range(X_test.shape[1]):#for all features
# 		normdist = npr.normal(stromameans[k],stromastddevs[k])
# 		temp.append(normdist)
# 	normprod=np.prod(temp)
# 	# print temp
# 	# print normdist
# 	stromaprob = normprod * epiprior / 112
# 	if epiprob>stromaprob:
# 		y_pred.append(1)
# 	elif stromaprob>epiprob:
# 		y_pred.append(2)


# print X_test.shape[1]
# onecount = 0
# twocount=0
# y_pred = []
# for i in range(len(x_test)):#for each row of test data
# 	#epiprob
# 	epiprob=epiprior
# 	temp=[]
# 		#find normal dist for all columns, calc product of all normal dists, mult by epiprior
# 	for j in range(x_test.shape[1]):#for all features
# 		epiprob *= 1.0/np.sqrt(2.0*np.pi*stromastddevs[j])
# 		epiprob *= np.exp((-0.5/stromastddevs[j])*(x_test[i][j]-stromameans[j])**2)
# 	#stromaprob
# 	stromaprob = stromaprior
# 	temp=[]
# 	for k in range(X_test.shape[1]):#for all features
# 		stromaprob *= 1.0/np.sqrt(2.0*np.pi*stromastddevs[k])
# 		stromaprob *= np.exp((-0.5/stromastddevs[k])*(x_test[i][k]-stromameans[k])**2)
# 		print stromaprob
# 	print stromaprob

# 	stromaprob = stromaprob/(stromaprob+epiprob)
# 	if epiprob>stromaprob:
# 		y_pred.append(1)
# 	elif stromaprob>epiprob:
# 		y_pred.append(2)
# 	else:
# 		y_pred.append(0)
# print y_pred
# gnb = GridSearchCV(GaussianNB(), lparameters)

gnb = GaussianNB()
gnb.fit(xdata,ydata)
y_pred = gnb.predict(X_test)

# epiprob=[]
# stromaprob=[]
# for i in range(len(y_pred)):
# 	epiprob.append(y_pred[i][0])
# 	stromaprob.append(y_pred[i][1])

# print epiprob
# print stromaprob

# correct = 0
# incorr = 0

# for i in range(len(x_test)):
# 	if y_pred[i]==y_test[i]:
# 		correct += 1
# 	else:
# 		incorr += 1

# print correct
# print incorr
# print 1.0*correct / (correct + incorr)
# print y_test
# print y_pred

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
np.savetxt('bayesout.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")

# test_header = "Id,EpiOrStroma"
# n_points = X_test.shape[0]
# # for i in range(n_points):
# # 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
# y_pred_pp = np.ones((n_points, 3))
# y_pred_pp[:, 0] = range(n_points)
# y_pred_pp[:, 1] = epiprob
# y_pred_pp[:, 2] = stromaprob

# np.savetxt('bayesout.csv', y_pred_pp, fmt='%f', delimiter=",",
#            header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.
    