###########################################
# Tools for Linear Regression models
#
# B. Stanley - Nov. 2015
#
# 1. Load data
#       - Loads a .txt file into
#         a numpy array.
#
# 2. Add Ones
#       - Adds a column of ones
#         to the input data array.
#
# 3. Normalise Features
#       - Performs feature scaling
#         so that each set of features
#         fall into a similar range.
#
# 4. Compute Cost
#       - Based on the input values (X),
#         the output values (y) and the
#         model parameters (theta).
#         Compute the error term, or
#         cost (J).
#
# 5. Gradient Descent
#       - Loads a .txt file into
#         a numpy array
###########################################

import numpy as np
import csv
import pdb

######## Load Data ########
def load_data(filename):
    
    # Loads arbitrtay data into numpy array of floating point numbers
    # first creates an appendable list, then converts to numpy array

    X = []
    
    with open(filename) as f:
        get = csv.reader(f, delimiter=',')
        for row in get:
            data, i = [], 0
            while i < len(row):
                data.append(row[i])
                i = i + 1
            X.append(data)

    X = np.asarray(X, dtype='float')
    return X

######## Add Column of Ones to Array ########

def addOnes(array):
    # Adds a column of ones to the data array
    
    ones  = np.ones((len(array),1))
    array = np.concatenate((ones,array), axis=1)
    
    return array

######## Normalise Features ########

def normalise_features(X):
    
    X_norm = np.asarray(np.zeros((len(X[:,0]),len(X[1,:]))))
    mu = []
    sigma = []
    
    for i in range(0, len(X[1])):
        
        feature = X[:,i]
        mu.append(np.mean(feature))
        sigma.append(np.std(feature))
        
        X_norm[:,i] = (feature - mu[i]) / sigma[i]
    
    return X_norm

######## Cost Function ########

def computeCost(X, y, theta):
    # Computes the sum of the squared errors after
    # the hypothesis function
    
    m = float(len(X)) # must be a float to ensure non-integer division
    
    hyp = np.sum(X * theta, axis=1)
    J = np.sum(  ((hyp - np.transpose(y)) * (hyp - np.transpose(y))),    axis=1);
    
    return J * (1/(2*m))

######## Gradient Descent ########

def gradientDescent(X, y, theta, alpha, iterations, min_iterations, early_stop):
    
    m = float(len(X))
    J_history = [[0],[0]]
    
    for i in range(0, iterations):
        
        hyp = np.sum(X *theta, axis=1)
        error = hyp - np.transpose(y)
        theta_change = (np.sum(X * np.transpose(error), axis=0) * alpha) * (1/m)
        theta = theta - np.transpose(theta_change);
        J_history.append(computeCost(X, y, theta))
        print "Cost = ", J_history[i]
        if i > min_iterations and (J_history[i-1] - J_history[i]) <= early_stop:
            break
    
    print "Number of iterations = ", i
    return theta, J_history