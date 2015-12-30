############################################
# Logistic Regression...
# A Python Implemtation of Andrew Ng's
# Machine Learning Class - Exercise 2
#
# B. Stanley Dec.2015
############################################

import logRegTools as lrt
import numpy as np
import pdb

# wrapper for optimize function
def objectiveFunction(initial_theta):
    return lrt.logComputeCost(X,y,initial_theta, lamda)

if __name__ == "__main__":
    
    print "Binary Classification -- a python implementation of Logistic Regression..."
    
    # Load and normalise data
    data = lrt.load_data('log_reg_data.txt')
    X_norm, mu, sigma = lrt.normalise_features(data[:, [0,1]])
    X = lrt.addOnes(X_norm)
    y = data[:, 2]

    # Initialise the model with zeros
    theta = np.zeros([1,3])

    # Initialise learning parameters
    max_iterations = 400
    lamda = 0.1

    print "Cost at inital theta (zeros) = ", lrt.logComputeCost(X,y,theta,lamda)
    print "Gradient at inital theta (zeros) = ", lrt.computeGradient(X,y,theta, lamda)
    raw_input("Press Enter to continue...")
    
    #train the model
    model = lrt.train(objectiveFunction, theta, max_iterations)

    print "model after optimisation", model
    print "Cost after optimisation = ", lrt.logComputeCost(X,y,model,lamda)
    print "Gradient after optimisation (zeros) = ", lrt.computeGradient(X,y,model,lamda)

    # Create new values for testing the model
    new_data = lrt.newDataExample(np.array([45, 85]), mu, sigma)

    print "Probability of student passing with 45 as first mark and 85 as second mark = ", lrt.predict(new_data, model)

