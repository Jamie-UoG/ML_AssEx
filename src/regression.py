#!/usr/bin/env python

import numpy as np


train_dir = "../training-data/"
test_dir = "../test-data/"
sub_dir = "../submissions/"

train_file_x = train_dir + "X_train.csv"
train_file_y = train_dir + "y_train.csv"
test_file = test_dir + "X_test.csv"


#-- Defines a model returning random results --#
def randomModel(test_args):

    # Just init random array of right size
    predictions = np.random.randint(0, 100, test_args.shape[0])
    return predictions


#-- Trains a model using the training data  --#
def train():

    # Get training data
    train_args = np.loadtxt(train_file_x, delimiter=',', skiprows=1)
    train_ans = np.loadtxt(train_file_y, delimiter=',', skiprows=1)[:,1]

    # Do training algorithm
    model = blankModel

    # Return model
    return model


#-- Runs a test and generates an answer for test data --#
def test(model):

    # Get test data
    test_args = np.loadtxt(test_file, delimiter=',', skiprows=1)

    # Run model on test data
    results = model(test_args)

    # Save test result to file
    saveResults(results)


#-- Saves the output of a test to csv file --#
def saveResults(results):

    # Header for output file
    header = "Id,PRP"
    n_points = results.shape[0]

    # Create new matrix for output data, fill with IDs and outputs
    output = np.ones((n_points, 2))
    output[:, 0] = range(n_points)
    output[:, 1] = results

    # Save to file
    np.savetxt(sub_dir+"my_submission.csv", output, fmt='%d', delimiter=",", header=header, comments="")
