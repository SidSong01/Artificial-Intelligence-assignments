# learning.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement linear and logistic regression
using the gradient descent method, as well as the binary perceptron algorithm. 
To complete the assignment, please modify the linear_regression(), binary_perceptron(), 
and logistic_regression() functions. 

The package `matplotlib` is needed for the program to run.
You should also use the 'numpy' library to vectorize 
your code, enabling a much more efficient implementation of 
linear and logistic regression. You are also free to use the 
native 'math' library of Python. 

All provided datasets are extracted from the scikit-learn machine learning library. 
These are called `toy datasets`, because they are quite simple and small. 
For more details about the datasets, please see https://scikit-learn.org/stable/datasets/index.html

Each dataset is randomly split into a training set and a testing set using a ratio of 8 : 2. 
You will use the training set to learn a regression model. Once the training is done, the code
will automatically validate the fitted model on the testing set.  
"""

# use math and/or numpy if needed
import math
import numpy as np

def linear_regression(x, y, logger=None):
    """
    Linear regression using full batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by x^T w, where x is a column vector. 
    The intercept term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    You should use as learning rate alpha=0.0001. If you scale the cost function by 1/#samples, use alpha=0.001  

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the target value for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
    
    Returns
    -------
    w: a 1D array
       linear regression parameters
    """
    max_iterations = 1000
    alpha = 0.0001
    N, f = np.shape(x)
    w = np.zeros(f)   

    for k in range(max_iterations): 
        pre_w = w
        prediction = np.dot(x,w)
        error = prediction - y
        '''
        both the cost and gradient can be divided by N for better comparison across different models,
        here didn't divide N for keeping the same output as the description video
        '''
        cost = np.dot(error, error) / 2
        gradientStep = np.dot(np.transpose(x), error)
        w = w - (alpha * gradientStep)
        if k % 10 == 0:
            logger.log(k, cost)
        if np.max(pre_w - w) < 1e-6:
            return w

    return w

def binary_perceptron(x, y, logger=None):
    """
    Binary classifaction using a perceptron. 
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by
        h = (x^T w) 
    with the decision boundary:
        h >= 0 => x in class 1
        h < 0  => x in class 0
    where x is a column vector. 
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    
    
    Parameters
    ----------
    x: a 2D array with the shape [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array with the shape [N]
       It is the ground truth value for each sample in x
    logger: a logger instance through which plotting loss
       Usage: Please do not use the logger in this function.
    
    Returns
    -------
    w: a 1D array
       binary perceptron parameters
    """
    N , f = np.shape(x)
    w = np.zeros(f)
    prediction = np.zeros(np.shape(y))
    counter = 0
    while not np.array_equal(prediction, y):
        prediction = np.dot(x, w)
        prediction = np.int64(prediction >= 0)
        w = w + np.dot(np.subtract(y, prediction), x)
        # just for visulization, the loss is meaningless
        loss = np.sum(np.subtract(y, prediction))
        logger.log(counter, abs(loss))
        counter += 1
            
    return w


def logistic_regression(x, y, logger=None):
    """
    Logistic regression using batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by p = sigmoid(x^T w)
    with the decision boundary:
        p >= 0.5 => x in class 1
        p < 0.5  => x in class 0
    where x is a column vector. 
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    In gradient descent, you should use as learning rate alpha=0.001    

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the ground truth label for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
        
    Returns
    -------
    w: a 1D array
       logistic regression parameters
    """

    max_iterations = 1000
    alpha = 0.0001
    N , f = np.shape(x)
    w = np.zeros((f))
    
    for k in range(max_iterations):
        pre_w = w
        prediction = np.dot(x, w)
        h = 1 / (1 + np.exp(-prediction))
        loss = h - y
        '''
        both the cost and gradient can be divided by N for better comparison across different models,
        here didn't divide N for keeping the same output as the description video
        '''
        cost = - (np.dot(y, np.log(h)) + np.dot((np.ones(np.shape(y)) - y), 
                                                np.log(np.ones(np.shape(h)) - h)))
        w = w - alpha * np.dot(np.transpose(x), loss)
        if k % 10 == 0:
            logger.log(k, cost)
        if np.max(pre_w - w) < 1e-6:
            return w
        
    return w

if __name__ == "__main__":
    import os
    import tkinter as tk
    from app.regression import App

    import data.load
    dbs = {
        "Boston Housing": (
            lambda : data.load("boston_house_prices.csv"),
            App.TaskType.REGRESSION
        ),
        "Diabetes": (
            lambda : data.load("diabetes.csv", header=0),
            App.TaskType.REGRESSION
        ),
        "Handwritten Digits": (
            lambda : (data.load("digits.csv", header=0)[0][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))],
                      data.load("digits.csv", header=0)[1][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))]),
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Breast Cancer": (
            lambda : data.load("breast_cancer.csv"),
            App.TaskType.BINARY_CLASSIFICATION
        )
     }

    algs = {
       "Linear Regression (Batch Gradient Descent)": (
            linear_regression,
            App.TaskType.REGRESSION
        ),
        "Logistic Regression (Batch Gradient Descent)": (
            logistic_regression,
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Binary Perceptron": (
            binary_perceptron,
            App.TaskType.BINARY_CLASSIFICATION
        )
    }

    root = tk.Tk()
    App(dbs, algs, root)
    tk.mainloop()
