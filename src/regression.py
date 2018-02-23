#!/usr/bin/env python

import numpy as np
from scipy.interpolate import *
from sklearn import linear_model
import math
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import *

from sklearn.preprocessing import PolynomialFeatures
train_dir = "../training-data/"
test_dir = "../test-data/"
sub_dir = "../submissions/"

train_file_x = train_dir + "X_train.csv"
train_file_y = train_dir + "y_train.csv"
test_file = test_dir + "X_test.csv"

# Get training data , only take PRP from answers
train_args = np.loadtxt(train_file_x, delimiter=',', skiprows=1)
train_ans = np.loadtxt(train_file_y, delimiter=',', skiprows=1)[:,1]

test_args = np.loadtxt(test_file, delimiter=',', skiprows=1)

# Run a full train/test cycle
# args:
#   am_test - amount of data to be designated for testing
def run(p_test, deg):

    am_test = p_test

    #print (str((train_ans.shape[0])) + "   numer of input elements.")
    #print ("Splitting input into blocks of " + str(am_test))

    # Full matrices of input data
    args = train_args.reshape(-1,6)
    ans = train_ans.reshape(-1,1)

    f_test_args = test_args.reshape(-1,6)

    avg_models = []

    if (p_test == 0):
        i = 1
        am_test = ans.shape[0]
        axis = range(0,ans.shape[0])
        plot(axis,ans.flatten(),'-')
    else:

        # Get number of iterations
        # If we have n number of test elements, and l total elements
        # We can do i = (l/n) iterations, taking the ith block of n elements in the input data each iteration
        length = ans.shape[0]
        i = math.floor(length/am_test)

        # Log answes to graph
        axis = range(0,ans.shape[0])
        plot(axis,ans.flatten(),'-')

    # Repeat the training and test for each iteration
    # Print RMSE for each iteration test
    # Plot test points
    final_ans = np.array([])

    for x in range(0, i):

        model = None

        # Matrix for the training independent variables

        f = train_args[(am_test*x):]
        s = train_args[:-(am_test*(x+1))]
        tr_args = np.concatenate((f,s))

        # Matrix for the training dependent variables
        f = train_ans[(am_test*x):]
        s = train_ans[:-(am_test*(x+1))]
        tr_ans = np.concatenate((f,s))

        # Train regression model
        model = train(tr_args, tr_ans, deg)

        avg_models += [model.coef_]

        # Matrix for the testing independent variables
        te_args = train_args[am_test*x:am_test*(x+1)].reshape(-1,6)

        # Matrix for the testing dependent variables
        te_ans = train_ans[am_test*x:am_test*(x+1)].reshape(-1,1)

        # Run model on test section of data
        answer = test(model, te_args, deg)
        final_ans = np.concatenate((final_ans, answer))

        # Log to graph
        ax = axis[am_test*x:am_test*(x+1)]
        plot(ax, answer.flatten(),'o')

    show()

    poly = PolynomialFeatures(degree=deg)
    te_ = poly.fit_transform(f_test_args)
    results = model.predict(te_)
    print (results)
    saveResults(results)
    plot(np.array(range(0,results.shape[0])), results.flatten(),'o')
    show()
    return
    # Finally, show
    # Print test performance
    printPerformnce(final_ans, ans, "done")

    # Create final model from average of all models
    final_model = np.zeros(avg_models[0].shape)
    for mod in avg_models:
        print (mod)
        final_model = final_model + mod
    print (final_model)
    print (len(avg_models))
    final_model = (final_model)/(len(avg_models))
    print (final_model)

    print (dir(model))
    print (model.get_params())

    model = linear_model.LinearRegression()
    model.intercept_=0
    model.coef_= final_model
    print (model.coef_)
    poly = PolynomialFeatures(degree=deg)
    te_ = poly.fit_transform(f_test_args)
    results = model.predict(te_)
    print (results)
    saveResults(results)
    plot(np.array(range(0,results.shape[0])), results.flatten(),'o')
    show()



def runAndTest(deg, print_bool):
    args = train_args.reshape(-1,6)
    ans = train_ans.reshape(-1,1)

    f_test_args = test_args.reshape(-1,6)

    model = train(args, ans, deg)

    answer = test(model, args, deg)

    print (math.sqrt(mean_squared_error(answer, ans)))

    if (print_bool):
        ax = range(0,ans.shape[0])
        plot(ax, ans.flatten(),'o')
        plot(ax, answer.flatten(),'-')
        show()

    answer = test(model, f_test_args, deg)

    if (print_bool):
        ax = range(0,f_test_args.shape[0])
        plot(ax, answer.flatten(),'-')
        show()

    saveResults(answer)



#-- Trains a model using the training data  --#
def train(tr_args, te_ans, deg):

    # PolynomialFeatures (prepreprocessing)
    poly = PolynomialFeatures(degree=deg)
    tr_ = poly.fit_transform(tr_args)

    # Perform linear regression on training data and get model
    ols = linear_model.LinearRegression(normalize=True)
    model = ols.fit(tr_, te_ans)

    # Print the model
    #print (model.coef_)

    # Return model
    return model


#-- Runs a test and generates an answer for test data --#
def test(model, te_args, deg):

    # Run model on test data
    poly = PolynomialFeatures(degree=deg)
    te_ = poly.fit_transform(te_args)
    results = model.predict(te_)

    return results


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


def printPerformnce(answer, te_ans, iter):
    rmse =  math.sqrt(mean_squared_error(answer, te_ans))

    # print ("ITERATION:   " + str(iter))
    # print ("\tOver " + str(answer.shape[0]) + " test entries:")
    # print ("\tRMSE    " + str(rmse))

    print(rmse)
    return rmse



for i in range(0,10):
    runAndTest(i, True)
