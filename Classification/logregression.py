#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif

min_max_scaler = preprocessing.MinMaxScaler()
lc_params = np.array([7.0])
lparameters = {'C': lc_params, 'penalty': ["l1","l2"]}

# logreg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=False, 
# 							intercept_scaling=1, class_weight='balanced', random_state=None, 
# 							solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, 
# 							warm_start=False, n_jobs=1)

# Load training and testing data
xdata = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
ydata = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

xdata = min_max_scaler.fit_transform(xdata)
X_test = min_max_scaler.fit_transform(X_test)

selector = SelectKBest(chi2, k = 32)
xdata = selector.fit_transform(xdata,ydata)
X_test = selector.transform(X_test)

print xdata.shape

# mutinfo = mutual_info_classif(xdata,ydata, n_neighbors=5)
# delarray=[]
# for i in range((len(mutinfo)-1),-1,-1):
# 	if mutinfo[i]<0.07:
# 		delarray.append(i)
# 		# print "DELET"
# print mutinfo		
# xdata = np.delete(xdata,delarray,axis=1)
# print xdata.shape
# xdata = VarianceThreshold(0.01).fit_transform(xdata)
# print xdata.shape

# print xdata.shape

#split data in half
# x_train,x_test=xdata[:len(xdata)/2],xdata[len(xdata)/2:]
# y_train,y_test=ydata[:len(ydata)/2],ydata[len(ydata)/2:]

#3/4 data for training
# x_train,x_test=xdata[:len(xdata)/4*3],xdata[len(xdata)/4*3:]
# y_train,y_test=ydata[:len(ydata)/4*3],ydata[len(ydata)/4*3:]

#1/4 data for training
# x_train,x_test=xdata[len(xdata)/4*3:],xdata[:len(xdata)/4*3]
# y_train,y_test=ydata[len(ydata)/4*3:],ydata[:len(ydata)/4*3]

#dont split the data
x_train = xdata
y_train = ydata



logreg = GridSearchCV(LogisticRegression(class_weight='balanced'), lparameters)

# n_samples=xdata.shape[0]
# scores = cross_val_score(logreg, xdata, ydata, cv=10, scoring='f1_macro')
# accuracy = scores.mean(), scores.std() * 2
# print accuracy

logreg.fit(x_train,y_train)
y_pred = logreg.predict(X_test) 
# score = logreg.score(x_test,y_test)
# print score

# epiprob=[]
# stromaprob=[]
# for i in range(len(y_pred)):
# 	epiprob.append(y_pred[i][0])
# 	stromaprob.append(y_pred[i][1])

# print epiprob
# print stromaprob

# print logreg.get_params(C)

# for i in range(len(y_test)):
# 	if y_test[i]==1:
# 		y_test[i]=0
# 	elif y_test[i] == 2:
# 		y_test[i]=1

# for i in range(len(y_pred)):
# 	if y_pred[i]==1:
# 		y_pred[i]=0
# 	elif y_pred[i] == 2:
# 		y_pred[i]=1

# scores = roc_auc_score(y_test,y_pred, average='macro')

# # print sorted(logreg.cv_results.keys())

# print scores

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

# epi=0
# stroma=0

# for j in range(len(y_pred)):
# 	if j == 0:
# 		continue;
# 	if y_pred[j]==1:
# 		epi += 1
# 	else:
# 		stroma+=1	

# print epi
# print stroma	

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
# for i in range(n_points):
# 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('logout.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")


# test_header = "Id,EpiOrStroma"
# n_points = X_test.shape[0]
# # for i in range(n_points):
# # 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
# y_pred_pp = np.ones((n_points, 3))
# y_pred_pp[:, 0] = range(n_points)
# y_pred_pp[:, 1] = epiprob
# y_pred_pp[:, 2] = stromaprob

# np.savetxt('logout.csv', y_pred_pp, fmt='%f', delimiter=",",
#            header=test_header, comments="")