[//]: # (Image References)

[image1]: ./example.png

## Overview

For the code to run, as in the previous assignment, you need to have numpy and matplotlib installed. After downloading the zip file, you should be able to run the code by typing the following at the command line:

`
$ python model.py
`

You will modify model.py to create neural network models. Even though the provided code can be used to solve a variety of machine learning tasks, you should focus on building neural networks for classification of handwritten digits.

![alt text][image1]

## Provided Code

To do so, you have been provided with a neural network mini-library (PNet) that you should import:

#### import PNet as nn

The library provides many optimized operations and data structures along with supporting common neural network functionality. In particular, instead of Python's built-in data types, please make use of the following:

* nn.Parameter to store a trainable parameter of a neural network such as the weight matrix of a layer, and the weights of the bias vector. All parameters should be 2-dimensional
* nn.Constant represents a matrix of floating point numbers. It should be used to store input features and target outputs/labels.
* nn.matmul to compute a matrix product between its inputs

#### Here is a list of activation functions that you can apply to activate the units of a hidden layer:

* nn.ReLU applies the element-wise Rectified Linear Unit nonlinearity: relu(x) = max(x, 0).
`Usage: nn.ReLU(z), which returns a node with the same shape as the input.`
* nn.sigmoid applies the element-wise sigmoid nonlinearity: sigmoid(x) = 1/(1+exp(-x)).

`Usage: nn.sigmoid(z), which returns a node with the same shape as the input.`
* nn.tanh applies the element-wise hyperbolic tangent nonlinearity: tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x)).

`Usage: nn.tanh(z), which returns a node with the same shape as the input.`

The library provides a number of objects that implement different loss (cost) functions including among others:

* nn.mse_loss computes a batched mean square loss, used for regression problems
`Usage: loss = nn.mse_loss(a, b), where a and b both have shape batch_size x num_outputs.`

* nn.softmax_cross_entropy_loss computes a batched softmax loss that can be used for classification problems
`Usage: [loss, probabilities] = nn.softmax_cross_entropy_loss(logits, labels), where logits have shape batch_size x num_classes, and the label have shape batch_size x 1. The term "logits" refers to the non-normalized scores produced by a model that will be converted to probabilities via the softmax activation function. The labels denote the actual classes of the batch data. Be sure not to swap the order of the arguments!`

The library provides a number of optimization algorithms that can be used to train the parameters of your model, including gradient descent:

* nn.sgd implements gradient descent updates.
`Usage: nn.sgd(params, stepsize), where params are the neural network parameters (w's and b's), and stepsize is the learning rate.`

Having set up an optimizer, one optimization iteration can be compute as follows:

* Perform backpropagation of the loss function to compute the gradients of the loss with respect to the parameters of the model

`Usage: loss.backward(), where loss is a loss node as defined above.`

* Update the parameters of the model based on the gradients

`Usage: optimizer.step(), where optimizer denotes the selected optimization algorithm.`

* Clear the currently computed gradients from the memory

`Usage: optimizer.zero_grad(), where optimizer denotes the selected optimization algorithm.`

When training a neural network, you will be passed a dataset object. The data is split into a training set, dataset.train, a validation set, dataset.validation, and a testing set, dataset.test. You will use the training set to learn the best parameters, the validation for hyperparameter tuning, and check the final performance of your network on the testing set.

#### Question 1 Digit Classification

`$ python model.py`

#### Question 2 Letter Classification

`$ python notMnist_model.py`
