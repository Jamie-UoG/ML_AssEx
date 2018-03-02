#!/usr/bin/env python

import numpy as np
from scipy.interpolate import *
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer

import math
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import *

from sklearn.preprocessing import PolynomialFeatures

minmax_scaler = preprocessing.MinMaxScaler()


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



'''
Runs a regression model and cross validation to give answer.
reg - type of regression model to use
scl - type of scaler to use
deg - polynomial degree to use
print - print out metrics and graphs
per - percent to use per segment (0 for 'leave 1 out validation')

'''
def runAndTestWithCrossValidation(reg, scl, deg, print_b, per):

    seg_n = int(per*(train_args.shape[0]))
    iters = float(train_args.shape[0])/float(seg_n)

    axis = range(0,train_args.shape[0])
    final_ans = np.array([])

    for i in range(0, int(iters)):

        # Segment our data
        if (i == int(iters-1)):
            tr_args = train_args[:(seg_n*i)]
            tr_ans = train_ans[:(seg_n*i)]

            te_args = train_args[(seg_n*i):]
            te_ans = train_ans[(seg_n*i):]
        else:

            f = train_args[:(seg_n*i)]
            s = train_args[(seg_n*(i+1)):]
            tr_args = np.concatenate((f,s))

            f = train_ans[:(seg_n*i)]
            s = train_ans[(seg_n*(i+1)):]
            tr_ans = np.concatenate((f,s))

            te_args = train_args[seg_n*i:seg_n*(i+1)]
            te_ans  = train_ans[seg_n*i:seg_n*(i+1)]

        # Do any scaler operations
        if (scl == "minmax"):
            tr_args = minmax_scaler.fit_transform(tr_args)
            te_args = minmax_scaler.fit_transform(te_args)
        elif (scl == "robust"):
            tr_args = preprocessing.robust_scale(tr_args)
            te_args = preprocessing.robust_scale(te_args)
        elif (scl == "log"):
            transformer = FunctionTransformer(np.log1p)
            tr_args = transformer.transform(tr_args)
            te_args = transformer.transform(te_args)


        # Convert to polynomial features
        poly = PolynomialFeatures(degree=deg)
        tr_args = poly.fit_transform(tr_args)
        te_args = poly.fit_transform(te_args)

        # Get our linear regression model
        if (reg == "normal"):
            model = train(tr_args, tr_ans)
        elif (reg == "lasso"):
            model = trainLasso(tr_args, tr_ans)
        elif (reg == "ridge"):
            model = trainRidge(tr_args, tr_ans)
        elif (reg == "elastic"):
            model = trainElastic(tr_args, tr_ans)
        else:
            return

        # Run test of validation segment
        answer = test(model, te_args)
        final_ans = np.concatenate((final_ans, answer))

        if (print_b):
            print ("For segment " + str(i))
            print ("    of size " + str(answer.shape[0]))
            print ("    RMSE of " + str(math.sqrt(mean_squared_error(answer, te_ans))))

    # Log to graph
    if (print_b):

        print ("Done all segments ")
        print ("    Total size    " + str(final_ans.shape[0]))
        print ("    Total RMSE of " + str(math.sqrt(mean_squared_error(final_ans, train_ans))))
        plot(axis, final_ans.flatten(),'o')
        plot(axis, train_ans.flatten(), 'ro')
        show()


'''
Runs a regression model and creates submission file.
reg - type of regression model to use
scl - type of scaler to use
deg - polynomial degree to use
print - print out metrics and graphs

'''
def runAndSubmit(reg, scl, deg, print_b):

    tr_args = train_args
    tr_ans = train_ans

    te_args = test_args

    # Do any scaler operations
    if (scl == "minmax"):
        tr_args = minmax_scaler.fit_transform(tr_args)
        te_args = minmax_scaler.fit_transform(te_args)
    elif (scl == "robust"):
        tr_args = preprocessing.robust_scale(tr_args)
        te_args = preprocessing.robust_scale(te_args)
    elif (scl == "log"):
        transformer = FunctionTransformer(np.log1p)
        tr_args = transformer.transform(tr_args)
        te_args = transformer.transform(te_args)


    # Convert to polynomial features
    poly = PolynomialFeatures(degree=deg)
    tr_args = poly.fit_transform(tr_args)
    te_args = poly.fit_transform(te_args)

    # Get our linear regression model
    if (reg == "normal"):
        model = train(tr_args, tr_ans)
    elif (reg == "lasso"):
        model = trainLasso(tr_args, tr_ans)
    elif (reg == "ridge"):
        model = trainRidge(tr_args, tr_ans)
    elif (reg == "elastic"):
        model = trainElastic(tr_args, tr_ans)
    else:
        return

    answer = test(model, tr_args)
    print (math.sqrt(mean_squared_error(answer, tr_ans)))

    if (print_b):
        ax = range(0, answer.shape[0])
        plot(ax, answer.flatten(),'o')
        plot(ax, tr_ans.flatten(),'ro')
        show()

    answer = test(model, te_args)

    if (print_b):
        ax = range(0, answer.shape[0])
        plot(ax, answer.flatten(),'o')
        show()
    saveResults(answer.flatten())


'''
Trains a model using training args and answers

train - regular linear regression
trainLasso - use lasso regression
trainRidge - use ridge regression
trainElastic - use elastic regression

'''
def train(tr_args, tr_ans):

    # Perform linear regression on training data and get model
    lin = linear_model.LinearRegression()
    model = lin.fit(tr_args, tr_ans)

    return model

def trainLasso(tr_args, tr_ans):

    # Perform lasso regression on training data and get model
    lass = linear_model.Lasso(alpha=1)
    model = lass.fit(tr_args, tr_ans)

    return model

def trainRidge(tr_args, tr_ans):

    # Perform ridge regression on training data and get model
    ridg = linear_model.Ridge(alpha=1)
    model = ridg.fit(tr_args, tr_ans)

    return model

def trainElastic(tr_args, tr_ans):

    # Perform elastic regression on training data and get model
    elas = linear_model.ElasticNet(alpha=1, l1_ratio=0.5)
    model = elas.fit(tr_args, tr_ans)

    return model


'''
Tests a given model with arguments to get predictions

'''
def test(model, te_args):

    # Run model on test data
    results = model.predict(te_args)

    return results


'''
Saves our result to the csv file

'''
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

runAndTestWithCrossValidation("elastic", "log", 3, True, 0.10)
runAndTestWithCrossValidation("elastic", "log", 3, True, 0.10)
runAndTestWithCrossValidation("elastic", "log", 2, True, 0.10)
