#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import optunity
import optunity.metrics

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif

# c_params = np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])

# parameters = {'kernel': ('linear','rbf','poly'), 'C': c_params, 'gamma': c_params, 'degree': np.array([2, 5]), 'coef0': np.array([1,2,3])}
                             	

supvecmac = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
 	probability=True, tol=0.001, cache_size=1000, class_weight='balanced', verbose=False,
  	max_iter=10000, decision_function_shape='ovr', random_state=None)
min_max_scaler = preprocessing.MinMaxScaler()

# Load training and testing data
xdata = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
ydata = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

# print "xdata before:"
# print xdata

xdata = min_max_scaler.fit_transform(xdata)
X_test = min_max_scaler.fit_transform(X_test)

selector = SelectKBest(f_classif, k = 10)
xdata = selector.fit_transform(xdata,ydata)
X_test = selector.transform(X_test)

# n_samples=xdata.shape[0]
# cv = LeaveOneOut()
# scores = cross_val_score(supvecmac, xdata, ydata, cv=cv)
# accuracy = scores.mean(), scores.std() * 2
# print accuracy

# cv_decorator = optunity.cross_validated(x=xdata, y=ydata, num_folds=5)


# def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0):
#     """A generic SVM training function, with arguments based on the chosen kernel."""
#     if kernel == 'linear':
#         model = sklearn.svm.SVC(kernel=kernel, C=C)
#     elif kernel == 'poly':
#         model = sklearn.svm.SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
#     elif kernel == 'rbf':
#         model = sklearn.svm.SVC(kernel=kernel, C=C, gamma=10 ** logGamma)
#     else:
#         raise ArgumentError("Unknown kernel function: %s" % kernel)
#     model.fit(x_train, y_train)
#     return model

# def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
#     model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
#     decision_values = model.decision_function(x_test)
#     return optunity.metrics.roc_auc(y_test, decision_values)

# svm_tuned_auroc = cv_decorator(svm_tuned_auroc)
# # svm_rbf_tuned_auroc(C=1.0, logGamma=0.0)

# optimal_svm_pars, info, _ = optunity.maximize_structured(svm_tuned_auroc, space, num_evals=150)
# print("Optimal parameters" + str(optimal_svm_pars))
# print("AUROC of tuned SVM: %1.3f" % info.optimum)


# epi=0
# stroma=0

# for j in range(len(ydata)):
# 	if j == 0:
# 		continue;
# 	if ydata[j]==1:
# 		epi += 1
# 	else:
# 		stroma+=1	

# print epi
# print stroma		

# print "xdata after"
# print xdata

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

# print ("x_train", x_train)
# supvecmac = GridSearchCV(svm.SVC(), parameters)

supvecmac.fit(x_train, y_train)
y_pred = supvecmac.predict(X_test)

# epiprob=[]
# stromaprob=[]
# for i in range(len(y_pred)):
# 	epiprob.append(y_pred[i][0])
# 	stromaprob.append(y_pred[i][1])

# print supvecmac.get_params()

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

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
# for i in range(n_points):
# 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('svmout.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")
