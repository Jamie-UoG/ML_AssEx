import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


# Load training and testing data
xdata = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
ydata = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

min_max_scaler = preprocessing.MinMaxScaler()
xdata = min_max_scaler.fit_transform(xdata)
X_test = min_max_scaler.fit_transform(X_test)

clf1 = RandomForestClassifier(n_estimators = 100,random_state=0, max_depth=10)
clf2 = GaussianNB()
clf3 = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
 	probability=True, tol=0.001, cache_size=1000, class_weight='balanced', verbose=False,
  	max_iter=10000, decision_function_shape='ovr', random_state=None)
clf4 = LogisticRegression(penalty='l2', dual=False, tol=0.0001,C=7.0, fit_intercept=False, 
							intercept_scaling=1, class_weight='balanced', random_state=None, 
							solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, 
							warm_start=False, n_jobs=1)
clf5 = XGBClassifier(max_depth=5, min_child_weight= 1, colsample_bytree=0.8,
			 subsample= 0.8,learning_rate= 0.1,n_estimators= 7, seed=0, 
             objective= 'binary:logistic')
clf6 = KNeighborsClassifier(n_neighbors=7)
lr = LogisticRegression(C=5)
sclf = StackingCVClassifier(classifiers=[clf1,clf2,clf6],use_probas=True,cv=6, meta_classifier=lr)

# params = {'kneighborsclassifier__n_neighbors':[5,7,8,9,10,11,12],
# 		'xgbclassifier__colsample_bytree':[0.7],
# 		'meta-logisticregression__C':[0.1,0.3,0.5,0.7,1.0,2.0,3.0,4.0,5.0,7.0,8.5,10.0],
# 		}

# dont split the data
x_train = xdata
y_train = ydata

# seed = 7
# test_size = 0.2

# x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=test_size, random_state=seed)


# grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5,refit=True,n_jobs=-1)
# grid.fit(x_train,y_train)

# cvkeys = {'mean_test_score', 'std_test_score','params'}

# for r, _ in enumerate(grid.cv_results_['mean_test_score']):
#     print("%0.3f +/- %0.2f %r"
#           % (grid.cv_results_[cvkeys[0]][r],
#              grid.cv_results_[cvkeys[1]][r] / 2.0,
#              grid.cv_results_[cvkeys[2]][r]))

# print('Best parameters: %s' % grid.best_params_)
# print('Accuracy: %.3f' % grid.best_score_)



for clf, label in zip([clf1,clf2,clf3,clf4,clf5,clf6,sclf],
						['Random Forest','Naive-Bayes','SVM',
						'Logistic Regression','xgBoost','KNN', 'StackingClassifier']):
	
	scores = cross_val_score(clf, x_train, y_train, cv=6, scoring='accuracy') 
	print("Accuracy: %0.4f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

sclf.fit(x_train,y_train)
y_pred = sclf.predict(X_test)

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
# for i in range(n_points):
# 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")