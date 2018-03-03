#!/usr/bin/env python
import numpy as np
import random as rand

import io

file1 = np.loadtxt('bayesout.csv', delimiter=',', skiprows=1)[:, 1]
file2 = np.loadtxt('svmout.csv', delimiter=',', skiprows=1)[:, 1]
file3 = np.loadtxt('logout.csv', delimiter=',', skiprows=1)[:, 1]

print file1

counter=0
ensemble = []
for i in range(len(file2)):
	# epi_proba = (file1[i][0] + file2[i][0] + file3[i][0])/3
	# stroma_proba = (file1[i][0] + file2[i][1] + file3[i][1])/3
	# print epi_proba
	# print stroma_proba
	# if epi_proba > stroma_proba:
	# 	ensemble.append(1)
	# elif stroma_proba > epi_proba:
	# 	print "T"
	# 	ensemble.append(2)
	# elif stroma_proba == epi_proba:
	# 	rand.randint(1,2)	

	if file1[i] == file2[i]:
		ensemble.append(file1[i])
	elif file1[i] == file3[i]:
		ensemble.append(file1[i])
	elif file2[i] == file3[i]:
		ensemble.append(file2[i])


print ensemble

y_pred=ensemble

test_header = "Id,EpiOrStroma"
n_points = file2.shape[0]
# for i in range(n_points):
# 	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=1))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
		 header=test_header, comments="")
