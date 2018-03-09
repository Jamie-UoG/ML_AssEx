import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif

#Handle Data
xdata = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
ydata = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

min_max_scaler = preprocessing.MinMaxScaler()
xdata = min_max_scaler.fit_transform(xdata)
X_test = min_max_scaler.fit_transform(X_test)

selector = SelectKBest(chi2, k = 30)
xdata = selector.fit_transform(xdata,ydata)
X_test = selector.transform(X_test)

for i in range(len(ydata)):
	if ydata[i]==1:
		ydata[i] = 0
	elif ydata[i]==2:
		ydata[i] = 1

# 1/10 data for training
# xdata,Xtest=xdata[int(len(xdata)/10):],xdata[:int(len(xdata)/10)]
# ydata,Y_test=ydata[int(len(ydata)/10):],ydata[:int(len(ydata)/10)]

#Paramaters for grid search
lc_params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10])
lparameters = {'C': lc_params, 'penalty': ["l2"]}
# logreg = GridSearchCV(LogisticRegression(class_weight='balanced'), lparameters)


#Cross validation using KFolds, returning accuracy and roc_auc
data_cv = KFold(n_splits=10)
data_cv.get_n_splits(xdata)
count=0
c=0.0
c2=0.0
avg_preds=[]
for train_i, test_i in data_cv.split(xdata):
	logreg = LogisticRegression(penalty='l2',C=6.6,class_weight='balanced')
	x_train, x_test = (xdata)[train_i], (xdata)[test_i]
	y_train, y_test = (ydata)[train_i], (ydata)[test_i]
	logreg.fit(x_train, y_train)
	print('Gridsearch: ',logreg.best_params_)
	y_preds = logreg.predict(X_test)
	# preds = cross_val_predict(logreg, x_train, y_train, cv=10)
	# print (preds)
		# avg_array += 

	# for i in range(len(logreg.coef_)):
	# 	if (count == 0):
	# 		avg_preds.append(logreg.coef_[i])
	# 	else:
	# 		logreg.coef_[i] += logreg.coef_[i]
	# print (avg_preds)
	scores = cross_val_score(logreg, x_train, y_train, cv=10)
	accuracy = scores.mean()
	scores2 = cross_val_score(logreg,x_train,y_train,cv=10, scoring='f1')
	AUC = scores2.mean()
	count += 1
	c = c + accuracy
	c2 = c2 + AUC
	print ("CV ",count,accuracy)
	print ("CV2",count,AUC)

y_pred=[]
for i in range(len(avg_preds)):
	if avg_preds[i]>5:
		y_pred.append(2)
	elif avg_preds[i]<6:
		y_pred.append(1)
	else:
		print ("SAME")

print (y_pred)		

avg = (c/10)
avg2 = (c2/10)
print ("average: ", avg)
print ("ROC: ", avg2) 

logreg.fit(xdata, ydata)
x_pred = logreg.predict(X_test)
for i in range(len(x_pred)):
	if x_pred[i]==0:
		x_pred[i] = 1
	elif x_pred[i]==1:
		x_pred[i] = 2
same=0
notsame=0

print (x_pred==y_pred)
for j in range(len(x_pred)):
	if x_pred[i]==y_pred[i]:
		same+=1
	else:
		notsame+=1

print ("same: ", same, "notsame: ", notsame)

# print('Gridsearch: ',logreg.best_params_)


# correct = 0
# incorr = 0
# for i in range(len(y_pred)):
# 	if y_pred[i]==Y_test[i]:
# 		correct += 1
# 	else:
# 		incorr += 1

# print (correct)
# print (incorr)
# print (1.0*correct / (correct + incorr))


# for i in range(len(y_pred)):
# 	if y_pred[i]==0:
# 		y_pred[i] = 1
# 	elif y_pred[i]==1:
# 		y_pred[i] = 2

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
# for i in range(n_points):
# 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")