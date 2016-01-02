############################################
# Linear Regression with Multiple Variables
# Python Implemtation of Andrew Ng's
# Machine Learning Class - Exercise 1
#
# B. Stanley Nov.2015
############################################

import numpy as np
import matplotlib.pyplot as plt
import csv
import pdb
import copy

import linRegTools as regression_tools

if __name__ == "__main__":
    
    # Load Data and Initial Values
    print "Loading Data..."
    Data = regression_tools.load_data('lin_reg_multi_data.txt')
    X = Data[:,[0,2]]
    y = Data[:,[2]]
    
    ######################################
    # Initial gradient descent parameters#
    ######################################
    
    iterations = 1000     
    min_iterations = 500
    early_stop = 0.05
    alpha = 0.1
    ######################################
    
    # Normalise data
    X = regression_tools.normalise_features(X)
    
    # Add a column of ones to X
    X = regression_tools.addOnes(X)
    
    # Initialise model (must be done after previous step)
    theta = np.zeros((1, len(X[1,:])))
    
    # Compute cost with initial theta
    cost = regression_tools.computeCost(X, y, theta)
    print "initial cost is", cost
    
    # Run gradient descent
    theta, J_history = regression_tools.gradientDescent(X, y, theta, alpha, iterations, min_iterations, early_stop)
    print "theta after gradient descent: ", theta

    plt.plot(J_history)
    plt.show()



