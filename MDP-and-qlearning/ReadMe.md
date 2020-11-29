[//]: # (Image References)

[image1]: ./example1.png
[image2]: ./example2.png
[image3]: ./result.png

### Second programming assignment of AI, license information has been included in files.

# Overview

Implementation of three classic algorithms for solving Markov Decision Processes: value iteration, policy iteration, and (tabular) Q-learning. 
Test these algorithms in simple grid-like worlds, and use Q-learning to control the crawler bot, making it learn to move forward.

![example][image1]
![example][image2]
---

# Requirements

* Python3

* matplotlib

```sh
pip install matplotlib
```


# TO DO:

* Value Iteration: There is a problem with how you are testing for convergence. Instead of abs(sum(v_pre - v)), you should be considering sum(abs(v_pre - v). 

* Policy Evaluation line 201: you shouldn't set the new value here. You should have a separate copy for the new values as you did in value iteration; otherwise certain states will be using updated values rather values from the previous step. 

* In Q-Learning, be careful about how the terminal state is handled. You might be resetting the environment (line 311) before updating the v and pi-values(line 315-316)

# Example Results

![alt text][image3]
