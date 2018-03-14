import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif, RFECV, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


#Handle Data
xdata = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
ydata = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

min_max_scaler = preprocessing.MinMaxScaler()
xdata = min_max_scaler.fit_transform(xdata)
X_test = min_max_scaler.fit_transform(X_test)

selector = SelectKBest(chi2, k = 23)
xdata = selector.fit_transform(xdata,ydata)
X_test = selector.transform(X_test)

for i in range(len(ydata)):
	if ydata[i]==1:
		ydata[i] = 0
	elif ydata[i]==2:
		ydata[i] = 1

#Paramaters for grid search
lc_params = np.array([6.5,6.6,6.7,6.8,6.9,7.0,7.1,7.2,7.3,7.4,7.5])
lparameters = {'C': lc_params, 'penalty': ["l2"]}
# logreg = GridSearchCV(LogisticRegression(class_weight='balanced'), lparameters)
logreg = LogisticRegression(penalty='l2',C=6.77,class_weight='balanced', fit_intercept=True)

#Cross validation using KFolds, returning accuracy and roc_auc
data_cv = KFold(n_splits=5)
data_cv.get_n_splits(xdata)
count=0
c=0.0
c2=0.0
c3=0.0
avg_preds=[]
for train_i, test_i in data_cv.split(xdata):
	x_train, x_test = (xdata)[train_i], (xdata)[test_i]
	y_train, y_test = (ydata)[train_i], (ydata)[test_i]
	logreg.fit(x_train, y_train)
	# print('Gridsearch: ',logreg.best_params_)
	y_pred = logreg.predict(x_test)
	testscores = accuracy_score(y_test, y_pred)
	scores = cross_val_score(logreg, x_train, y_train, cv=5)
	accuracy = scores.mean()
	scores2 = cross_val_score(logreg,x_train,y_train,cv=5, scoring='roc_auc_score')
	AUC = scores2.mean()
	count += 1
	c = c + accuracy
	c2 = c2 + AUC
	c3 = c3 + testscores

	print ("ACC ",count,accuracy)
	print ("ROC ",count,AUC)
	print ("Test",count,testscores)

avg = (c/5)
avg2 = (c2/5)
avg3 = (c3/5)

print ("average: ", avg)
print ("ROC: ", avg2) 
print ("Test: ", avg3) 

logreg.fit(xdata, ydata)
print (X_test.shape)

y_pred = logreg.predict(X_test)
for i in range(len(y_pred)):
	if y_pred[i]==0:
		y_pred[i] = 1
	elif y_pred[i]==1:
		y_pred[i] = 2

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
# for i in range(n_points):
# 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")