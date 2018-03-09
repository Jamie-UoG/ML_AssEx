import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# min_max_scaler = preprocessing.MinMaxScaler()

# Load training and testing data
xdata = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
ydata = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

for i in range(len(ydata)):
	if ydata[i]==1:
		ydata[i]=0
	elif ydata[i] == 2:
		ydata[i]=1

# var_selector = VarianceThreshold(0.01)
# xdata = var_selector.fit_transform(xdata)
# X_test = var_selector.transform(X_test)
# print (xdata.shape)

# xdata = min_max_scaler.fit_transform(xdata)
# X_test = min_max_scaler.fit_transform(X_test)

# 1/4 data for training
# x_train,x_test=xdata[int(len(xdata)/4*3):],xdata[:int(len(xdata)/4*3)]
# y_train,y_test=ydata[int(len(ydata)/4*3):],ydata[:int(len(ydata)/4*3)]

##dont split the data
x_train = xdata
y_train = ydata

# 3/4 data for training
# x_train,x_test=xdata[:int(len(xdata)/4*3)],xdata[int(len(xdata)/4*3):]
# y_train,y_test=ydata[:int(len(ydata)/4*3)],ydata[int(len(ydata)/4*3):]


# cv_params = {}
# ind_params = {'max_depth': 5, 'min_child_weight': 1, 
# 			 'subsample': 0.9,'learning_rate': 0.1,'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
#              'objective': 'binary:logistic'}
# model = GridSearchCV(xgb.XGBClassifier(**ind_params), 
#                             cv_params, 
#                              scoring = 'accuracy', cv = 8, n_jobs = -1) 


seed = 5
test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=test_size, random_state=seed)
deval = xgb.DMatrix(x_test,y_test)
watchlist = [(deval, 'eval')]


xgdmat = xgb.DMatrix(x_train,label=y_train)
xtmat = xgb.DMatrix(X_test)

my_params = {'max_depth': 5, 'min_child_weight': 1, 
			'subsample': 0.4,'eta': 0.1,'silent': 1, 'colsample_bytree': 0.8 , 
            'objective': 'binary:logistic', 'eval_metric':'error','seed':5}
n_folds = 5
early_stopping = 1000
xcv = xgb.cv(params=my_params, dtrain=xgdmat, num_boost_round=5000, nfold=10, early_stopping_rounds=early_stopping,verbose_eval=True)


final_gb=xgb.train(my_params,xgdmat,70,evals=watchlist)


y_pred = final_gb.predict(xtmat)
importance= final_gb.get_fscore()
print(xcv)


# cv_xgb = xgb.cv(params=my_params, dtrain=xgdmat, num_boost_round=100000, nfold=5,
# 		metrics=['error'],early_stopping_rounds=1000, verbose_eval=True)

# print (cv_xgb)

# best_round = cv_xgb.shape[0] - 1000
# best_round = int(best_round/0.8)

# print (best_round)


# model = XGBClassifier(max_depth=5, min_child_weight=1, subsample=0.9,learning_rate=0.1,n_estimators=29)
# model.fit(x_train,y_train)


pred = [(round(value)) for value in y_pred]

# score = accuracy_score(y_test, pred)
# print ("score %.2f%%" % (score*100.0))

y_pred = [(value+1) for value in pred]


# print (model.best_params_)
# xgb.plot_importance(model)

# print (y_pred)

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
# for i in range(n_points):
# 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")


