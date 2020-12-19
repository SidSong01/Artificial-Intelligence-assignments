## To Do:

Observation update : Here, you say that the true distance follows a Gaussian with mean around the observed distance. It should be the other way around, i.e. the observed distance follows a Gaussian centered at the true distance, i.e. normal_pdf(dist, App.SENSOR_NOISE, observed_distances[i]). Note that both are equivalent but conceptually the latter makes more sense.

[//]: # (Image References)

[image1]: ./example.png
[image2]: ./instructions.PNG

## Overvies

To ensure the safety of its tigers, a zoo asked you to install a localization system that can track the activities of the tiger agents. You decided to utilize your knowledge about Hidden Markov Models and implement two inference algorithms: an exact algorithm that computes a full probability distribution over an agent's location, and particle filtering which approximates the same distribution using a set of samples. 

![alt text][image1]

## Basic Requirements

You should try to take advantage of Python's math and random libraries. 

`
$ python hmm.py
`

![alt text][image2]

## Important notes: 

You can add more tiger agents to the environment by changing the #agents parameters while calling "App(#agents, algs, root)" at the end of the file. If your code works with a single agent, it should be able to scale to any number of agents since they are treated independently.
Please read carefully all the comments in the hmm.py file. It's unlikely you can start implementing the two algorithms without getting a better idea of how the code works and what is expected from each function.
We strongly recommend that you watch/review the lectures on HMMs and Particle Filtering before getting started. The questions are short and straightforward -- no more than about 70 lines of code in total -- but only if your understanding of the probability and inference concepts is clear!
