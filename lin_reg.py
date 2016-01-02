############################################
# Linear Regression with One Variable
# Python Implemtation of Andrew Ng's
# Machine Learning Class - Exercise 1
#
# B. Stanley Nov.2015
############################################


import numpy as np
import matplotlib.pyplot as plt
import csv
import pdb

import linRegTools as regression_tools

######## Plot Data ########

def plot_data(X, y):
    
    # Plots a 2D scatter graph of the
    # data matrix
    
    plt.plot(X, y, 'ro')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

############################

if __name__ == "__main__":
    
    #Load Data and Initial Values
    print "Loading Data..."
    Data = regression_tools.load_data('lin_reg_data.txt')
    X = Data[:,[0]]
    y = Data[:,[1]]
    theta = np.zeros((1,2))
    
    #Initial gradient decsent parameters
    iterations = 1500
    min_iterations = 500
    early_stop = 0.5
    alpha = 0.01
    
    # Add a column of ones to X
    X = regression_tools.addOnes(X)

    cost = regression_tools.computeCost(X, y, theta)
    print "Cost at inital theta (zeros) = ", cost
    
    raw_input("Press Enter to continue...")

    theta, J_history = regression_tools.gradientDescent(X, y, theta, alpha, iterations, min_iterations, early_stop)
    print "theta after gradient descent: ", theta

    #plt.plot(J_history)
    plot_data(X[:,(1)], y)
    #todo -- Add hypothesis function to plot
    plt.show()



