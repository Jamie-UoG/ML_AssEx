#!/usr/bin/env python
import numpy as np
import math
from sklearn.metrics import mean_squared_error
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


'''
Runs the regression model and cross validation to give answer.
train_args - training data arguments
train_ans - training data answers
per - percent to use per segment as float
'''
def runAndTestWithCrossValidation(train_args, train_ans, per):

    seg_n = int(per*(train_args.shape[0]))
    iters = float(train_args.shape[0])/float(seg_n)

    axis = range(0,train_args.shape[0])
    final_ans = np.array([])
    for i in range(0, int(iters)):

        # Segment our data
        # If last iteration use all remaining data
        if (i == int(iters-1)):
            tr_args = train_args[:(seg_n*i)]
            tr_ans = train_ans[:(seg_n*i)]

            te_args = train_args[(seg_n*i):]
            te_ans = train_ans[(seg_n*i):]
        # Else use ith segment for testing
        else:

            f = train_args[:(seg_n*i)]
            s = train_args[(seg_n*(i+1)):]
            tr_args = np.concatenate((f,s))

            f = train_ans[:(seg_n*i)]
            s = train_ans[(seg_n*(i+1)):]
            tr_ans = np.concatenate((f,s))

            te_args = train_args[seg_n*i:seg_n*(i+1)]
            te_ans  = train_ans[seg_n*i:seg_n*(i+1)]


        # Perform linear regression on training data and get model
        model = train(tr_args, tr_ans)

        # Run test of validation segment
        answer = model.predict(te_args)

        # Append answer
        final_ans = np.concatenate((final_ans, answer))

    # Print rmse of combined answers
    rmse = math.sqrt(mean_squared_error(final_ans, train_ans))
    print ("RMSE of combined predictions during cross validation is:")
    print ("\t" + str(rmse))
    print ("\n")



'''
Runs the regression model and creates a submission file.
train_args - training data arguments
train_ans - training data answers
test_args - testing data arguments
'''
def runAndSubmit(train_args, train_ans, test_args):

    # Perform linear regression on training data and get model
    model = train(train_args, train_ans)

    # Predict with test arguments
    answer = model.predict(test_args)

    # Header for output file
    header = "Id,PRP"
    n_points = answer.shape[0]

    # Create new matrix for output data, fill with IDs and outputs
    output = np.ones((n_points, 2))
    output[:, 0] = range(n_points)
    output[:, 1] = answer

    # Save to file
    np.savetxt(sub_dir+"my_submission.csv", output, fmt='%d,%f', delimiter=",", header=header, comments="")
    print ("Testing predictions saved to file.")



'''
Trains and returns a regression model.
train_args - training data arguments
train_ans - training data answers
'''
def train(train_args, train_ans):
    lin = linear_model.LinearRegression()
    model = lin.fit(X_train, y_train)
    return model


runAndTestWithCrossValidation(X_train, y_train, 0.125)
runAndSubmit(X_train, y_train, X_test)
