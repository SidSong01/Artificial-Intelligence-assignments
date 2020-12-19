## To Do:

Observation update : Here, you say that the true distance follows a Gaussian with mean around the observed distance. It should be the other way around, i.e. the observed distance follows a Gaussian centered at the true distance, i.e. normal_pdf(dist, App.SENSOR_NOISE, observed_distances[i]). Note that both are equivalent but conceptually the latter makes more sense.


## Overvies

To ensure the safety of its tigers, a zoo asked you to install a localization system that can track the activities of the tiger agents. You decided to utilize your knowledge about Hidden Markov Models and implement two inference algorithms: an exact algorithm that computes a full probability distribution over an agent's location, and particle filtering which approximates the same distribution using a set of samples. 

## Basic Requirements

You should try to take advantage of Python's math and random libraries. 

`sh
python hmm.py
`

Your need to modify hmm.py in order to implement two HMM inference algorithms. To simplify things, we assume that the zoo is a discretized 2D rectangular grid equipped with a number of landmarks that facilitate the tracking of the agent. At each time step t, let Xt denote the the actual location of an agent (which is unobserved). We assume there is a local conditional distribution p(xt | xt-1) that governs the agent's movement. In addition, we receive as measurement Et a vector of four values that denote the agent's distance to each of the four landmarks that are installed in the four corners of the environment. However, the distance observations are noisy. For each landmark i, the distance-based sensor provides a measurement \tiny E^i_t, which is a Gaussian random variable with mean equal to the true distance between the landmark positioned at Li and the agent, and variance σ2, i.e.
