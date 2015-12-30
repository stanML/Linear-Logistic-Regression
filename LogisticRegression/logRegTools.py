###########################################
# Tools for Logistic Regression model
#
# B. Stanley - Nov. 2015
#
# 1. Load data
#       - Loads a .txt file into
#         a numpy array.
#
# 2. Add Ones
#       - Adds a column of ones
#         to the input data (array).
#
# 3. Normalise Features
#       - Performs feature scaling
#         so that each set of features
#         fall into a similar range.
#
# 4. Sigmoid function
#        - Returns a "close to binary" value
#          after applying the sigmoid funtion
#          to the input.
#
# 4. Regularised Logistic Cost Function
#       - Based on the input values (X),
#         the output values (y) and the
#         model parameters (theta).
#         Compute the error term, or
#         cost (J). Include regularistion
#         parameter (lamda), to prevent
#         over fitting.
#
# 4. Compute gradients with regularisation
#       - Based on the input values (X),
#         the output values (y) and the
#         model parameters (theta) and
#         regularisation parameter (lamda),
#         Compute the gradients.
#
# 6. Optimise model
#       - Uses the "fmin_bfgs" optimisation
#         function from the SciPy library to
#         calcualte optimum model parameters.
#
# 7. Predict
#       - Given the learned parameters (model)
#         and a single data example (X), return
#         the probability of a binary result.
#
# 8. New data example
#       - Given a new example (data), perform
#         normalisation using the mean (mu)
#         and standard deviation (sigma) of the
#         training data and add an intercept
#         term to the array.
#
###########################################

import numpy as np
import csv
import pdb
from scipy.optimize import fmin_bfgs

######## Load Data ########
def load_data(filename):
    
    # Loads arbitrtay .txt file into numpy array of floating point numbers
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

# Adds a column of ones to the data array

def addOnes(array):
    
    ones  = np.ones((len(array),1))
    array = np.concatenate((ones,array), axis=1)
    
    return array

######## Normalise Features ########

# Performs feature scaling to improve the efficiency of
# learning algorithm

def normalise_features(X):
    
    X_norm = np.asarray(np.zeros((len(X[:,0]),len(X[1,:]))))
    mu = []
    sigma = []
    
    for i in range(0, len(X[1])):
        
        feature = X[:,i]
        mu.append(np.mean(feature))
        sigma.append(np.std(feature))
        
        X_norm[:,i] = (feature - mu[i]) / sigma[i]
    
    return X_norm, mu, sigma

######## Sigmoid Function ########

# using a lambda function to map sigmoid
# to each value in the input array

def sigmoid(a):
    
    if len(a) == 1:
        g = 1 / (1 + np.exp(-a))

    else:
        g = map(lambda x : 1 / (1 + np.exp(-x)), a)

    return g

######## Logistic Cost Function ########

# Computes the sum of the squared errors after
# the hypothesis function

def logComputeCost(X, y, theta, lamda):
    
    m = float(len(X)) # must be a float to ensure non-integer division
    hyp = np.array(sigmoid(np.sum(X * theta, axis=1))) # sigmoid of the input after the model is applied

    J = np.sum(-y * np.log(hyp)) - np.sum((1 - y) * (np.log(1 - hyp)))
    J = J * (1/m)
    
    # Compute the regularisation term
    reg = (lamda / float(2*m)) * np.sum(theta**2)
    J = J + reg

    return J

######## Compute Gradients ########

# Computes the gradient of the model based on the parameters (theta)

def computeGradient(X,y,theta,lamda):

    m = float(len(X)) # must be a float to ensure non-integer division
    hyp = np.array(sigmoid(np.sum(X * theta, axis=1))) # sigmoid of the input after the model is applied
    
    # Compute the regularisation term
    reg = (lamda / float(m)) * theta
    
    return ((1/m)* np.sum(((hyp - y) * np.transpose(X)), axis=1)) + reg

######## Optimise Model ########

# Using an optimisation algorithm from the SciPy library
# return the parameters that find the minimum of the
# objective (cost) function.

def train(objectiveFunction, theta, iterations):
    
    model =  fmin_bfgs(objectiveFunction, theta, maxiter=iterations, disp=0)

    return model

######## Predict ########

def predict(X, model):

    # sigmoid of the input after the model is applied
    hyp = np.array([np.sum(X * model, axis=0)])
    output = sigmoid(hyp)

    return output

######## New Data Example ########

# Prepare new data as an input to the learnt model

def newDataExample(data, mean, standard_dev):
    
    prep = np.asarray(np.zeros(data.shape))
    
    for i in range(0, len(data)):
        
        feature = data[i]
        prep[i] = (feature - mean[i]) / standard_dev[i]
    
    return np.concatenate((np.array([1]), prep), axis=0)

