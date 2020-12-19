## To Do:

For perceptron training, you should sequentially check every datapoint and update the weights for each training row.

[//]: # (Image References)

[image1]: ./example.png

## Overview

You will implement linear regression, logistic regression, and binary perceptron for supervised learning problems. You will test your algorithms on some simple, toy-like, data sets extracted from the scikit-learn machine learning library. These data sets are useful to quickly illustrate the behavior of traditional supervised learning algorithms and help you better understand the topics covered in class. Each data set is randomly shuffled and the data is then split into a training and testing set. You will use the training set to learn the best model (set of weights) and then check its performance on the testing set.

![alt text][image1]

## Basic Requirements

For the code to run, you need to have matplotlib installed. You should also try to take advantage of Python's numpy library. After downloading the zip file, you should be able to display the GUI for the homework by typing the following at the command line:

`
$ python learning.py
`

Your task is to modify learning.py in order to implement the objective functions and gradient calculations for linear and logistic regression, as well as the binary perceptron algorithm.

#### Question 1 Linear Regression

Modify the linear_regression function in learning.py to implement linear regression using full-batch gradient descent. The goal here is to find the best choice of weights wi for predicting the target variable y as a linear function of input features x. To do so, you are going to use the sum of squared errors as the loss/cost function, and minimize it using batch gradient descent with a fixed learning rate.

You should first test your implementation on the Boston housing data set which contains housing values in suburbs of Boston. This data set is a copy of the UCI ML housing dataset. The y-values are the prices of the houses in $1K, and the x-values correspond to 13 input feature variables (see here for details). Overall, there are 506 examples in the data set, split into a training and testing set. The objective is to predict the value of a house using the given features. Once your code is working, you can also test it against the diabetes data set which contains a quantitative measure of disease progression for 442 diabetes patients based on ten variables (see here for details).

#### Question 2 Binary Perceptron

Modify the binary_perceptron function in learning.py to solve binary classification problems using a binary perceptron. The goal here is to find the weights wi for predicting the correct binary label y given input features x. To do so, you should repeatedly loop over the training examples, compute the dot product between the weight vector and each given example, and update w for examples that are misclassified. When an entire pass over the training set is completed without making any mistakes, training can terminate.

You should use your code in the handwritten digits dataset to learn to classify images of digits as either “0” or “1”. This dataset is a reduced version of the MNIST dataset consisting of 359 examples; each of the digits is represented by an 8x8 grid of pixel values, and each pixel is an integer in the range [0,16]. If your code is correct, your classifier should be able to achieve 100% accuracy on the training set. The reason is that 0 and 1 digits look significantly different and the resulting training data sets are perfectly separable allowing the algorithm to converge.

Unfortunately, when the training data is not separable, the weights might thrash and the perceptron may not converge. For example, depending on the training data extracted from the breast cancer dataset, the algorithm may never converge. In addition, the perceptron suffers from mediocre generalization as it doesn't care for finding the optimal w but rather focuses on computing a weight vector that can simply separate the two classes. In the handwritten digits classification problem, for example, the perceptron can result in a testing accuracy smaller than 100%, despite the simplicity of the problem. To address these issues, you will implement logistic regression next.

#### Question 3 Logistic Regression
Modify the logistic_regression function in learning.py to implement logistic regression using full-batch gradient descent for solving binary classification problems. The goal here is to find the best choice of weights w for predicting the correct binary label y given input features x. To do so, you are going to use the negative log-likelihood estimate for w as the loss/cost function, and minimize it using batch gradient descent with a fixed learning rate. Note that as gradient descent for linear and logistic regression differ only in the function that your are learning (linear vs sigmoid), implementing logistic regression should be pretty straightforward.

You should use your code in the handwritten digits dataset. If your code is correct, your classifier should be able to achieve 100% accuracy both on the training and testing sets. Similarly, you should be able to obtain high classification accuracy in the breast cancer dataset, where the task here is to predict whether a tumor is malignant or benign based on 30 input features. The dataset consists of 569 examples, and the features are computed from a digitized image of a fine needle aspirate of a breast mass (see here for details).

## Important notes: 

1. An extra ‘1’ feature is automatically added to each data set so that w0 will always act as an intercept term in linear regression, and as a bias term in the logistic regression.

2. During the training, the performance of your model can be displayed by calling logger.log(i, loss) where i is the number of gradient descent iterations, and loss denotes the cost function that you are minimizing. This can help you debug your implementation and see the convergence rate of your algorithms. Once training is done, the performance of your learned model on the testing set is automatically displayed.

3. As different data sets can have different number of examples, m, for better comparison across different models, you may want to consider dividing the cost function by m to ensure that the size of the training data set doesn't affect the function. This means that while updating the weights, the gradient should also be divided by m.

4. In implementing batch gradient descent for linear and logistic regression, having for-loops where weights are computed one by one can work fine in small toy-like data sets. However, this approach is too slow for large problems that are more interesting. To address this issue, you should vectorize your gradient update implementation. To do so, you can use numpy's array structure to store the w, x, and y variables, and take advantage of numpy's dot and/or matmul functions to efficiently compute each iteration of gradient descent.

5. As you will be using a fixed learning rate to do the gradient descent, the only parameter to tune is the number of gradient descent iterations. However, you may want to exit your algorithm early if the values have already converged. To do so, at each iteration you can compute the absolute difference between current and previous weights. If the max over these differences is below some small user-specified threshold, you can quit the algorithm. Alternatively, you can check the infinity norm of your gradient for convergence.
