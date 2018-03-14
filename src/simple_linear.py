#!/usr/bin/env python
import numpy as np
from sklearn import linear_model


#-- Directories and files --#
train_dir = "../training-data/"
test_dir = "../test-data/"
sub_dir = "../submissions/"
train_file_x = train_dir + "X_train.csv"
train_file_y = train_dir + "y_train.csv"
test_file = test_dir + "X_test.csv"


#-- Load training and testing data --#
X_train = np.loadtxt(train_file_x, delimiter=',', skiprows=1)
y_train = np.loadtxt(train_file_y, delimiter=',', skiprows=1)[:,1]
X_test = np.loadtxt(test_file, delimiter=',', skiprows=1)


#-- Perform linear regression on training data and get model --#
lin = linear_model.LinearRegression()
model = lin.fit(X_train, y_train)


#-- Run model on test data --#
results = model.predict(X_test)


#-- Save to submission file --#
# Header for output file
header = "Id,PRP"
n_points = results.shape[0]

# Create new matrix for output data, fill with IDs and outputs
output = np.ones((n_points, 2))
output[:, 0] = range(n_points)
output[:, 1] = results

# Save to file
np.savetxt(sub_dir+"my_submission.csv", output, fmt='%d,%f', delimiter=",", header=header, comments="")
