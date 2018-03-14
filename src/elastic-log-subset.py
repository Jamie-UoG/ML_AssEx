#!/usr/bin/env python

import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures

# Config
deg = 3             # Degree of regression
alp = 5             # Alpha - constant that multiplies penalty terms
l1r = 0.5           # Level - elastic net mixing parameter
max_iter = 100000   # Max number of iterations for elastic net

# Directories and files
train_dir = "../training-data/"
test_dir = "../test-data/"
sub_dir = "../submissions/"
train_file_x = train_dir + "X_train.csv"
train_file_y = train_dir + "y_train.csv"
test_file = test_dir + "X_test.csv"


# Load training and testing data
X_train = np.loadtxt(train_file_x, delimiter=',', skiprows=1)
y_train = np.loadtxt(train_file_y, delimiter=',', skiprows=1)[:,1]
X_test = np.loadtxt(test_file, delimiter=',', skiprows=1)


# Take subset of training and test features
# Use sum of max/min cache and memory
data = X_train
new_ca = np.sum((data[:,4],data[:,5]), axis=0)
new_mem = np.sum((data[:,1],data[:,2]), axis=0)
new_data = np.column_stack((data[:,0] , new_mem, data[:,3], new_ca))
X_train = new_data

data = X_test
new_ca = np.sum((data[:,4],data[:,5]), axis=0)
new_mem = np.sum((data[:,1],data[:,2]), axis=0)
new_data = np.column_stack((data[:,0] , new_mem, data[:,3], new_ca))
X_test = new_data


# Perform input scaling with log1p (log plus 1)
transformer = FunctionTransformer(np.log1p)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)


# Convert to polynomial features
poly = PolynomialFeatures(degree=deg)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)


# Perform elastic regression on training data and get model
elas = linear_model.ElasticNet(alpha=alp, l1_ratio=l1r, max_iter=max_iter)
model = elas.fit(X_train, y_train)


# Run model on test data
results = model.predict(X_test)


# Header for output file
header = "Id,PRP"
n_points = results.shape[0]


# Create new matrix for output data, fill with IDs and outputs
output = np.ones((n_points, 2))
output[:, 0] = range(n_points)
output[:, 1] = results


# Save to file
np.savetxt(sub_dir+"my_submission.csv", output, fmt='%d,%f', delimiter=",", header=header, comments="")
