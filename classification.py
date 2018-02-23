#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

#split data in half
#X_train,X_test=xdata[:len(xdata)/2],xdata[len(xdata)/2:]
#y_train,ytest=ydata[:len(ydata)/2],ydata[len(ydata)/2:]

#3/4 data for training
#X_train,X_test=xdata[:len(xdata)/4*3],xdata[len(xdata)/4:]
#y_train,ytest=ydata[:len(ydata)/4*3],ydata[len(ydata)/4:]

# Fit model and predict test values
classes = {1:'ro',2:'bo'}
plt.figure()
for cl in classes:
    pos = np.where(y_train == cl)[0]
    plt.plot(X_train[pos,0],X_train[pos,1],classes[cl])

#plt.show()
y_pred = np.random.randint(1, 3, X_test.shape[0])

def knn_classifier(X_train,y_train,testrow,K=3):
    distances = ((X_train - testrow)**2).sum(axis=1)
    dc = zip(distances,y_train)
    dc = sorted(dc,key = lambda x:x[0])
    classes = []
    votes = []
    for k in range(K):
        this_class = dc[k][1] # get the class of the kth ranked
        if not this_class in classes:
            classes.append(this_class)
            votes.append(1)
        else:
            index = classes.index(this_class)
            votes[index] += 1
    best_class = classes[0]
    best_vote = votes[0]
    pos = 1
    for cl in classes[1:]:
        if votes[pos] > best_vote:
            best_vote = votes[pos]
            best_class = cl
        pos += 1
    return best_class

'''
Kvals = np.arange(1,300,2)
accuracy = []
for k in Kvals:
    correct = 0
    for i,row in enumerate(X_test):
        c = knn_classifier(X_train,y_train,row,K=k)
        if c == ytest[i]:
            correct += 1
    accuracy.append(1.0*correct / (1.0*len(X_test)))

plt.figure()
plt.plot(Kvals,accuracy)
plt.show()
'''

y_pred = []

# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
for i in range(n_points):
	y_pred.append(int(knn_classifier(X_train,y_train,X_test[i-1,:],K=3))) 
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.
    