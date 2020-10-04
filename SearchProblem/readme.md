[//]: # (Image References)

[image1]: ./example.png

### First programming assignment of AI, license information has been included in files.

# Overview

Implementation of a number of search algorithms to guide an agent through a grid-based world. For simplicity, we'll assume that the state space is 2-dimensional, consisting of the grid location of the agent.

![UNF example][image1]
---

# TO USE

```sh
python search.py
```

Then refer to the UI.

# Basic Requirements
The code for this project consists of two Python files. Python3 is required.

* `search.py` is the main execution of the project.
* `search_app.py` is the necessary data structure and functions.

Your task is to modify search.py in order to implement three of the search algorithms covered in class: depth-first search, uniform cost search, and A*. 

----

* Depth First Search

The depth_first_search function in `search.py`. You should employ the graph search version of DFS which allows a state to be expanded only once. Test your code on different maps to evaluate its correctness. We've provided a number of default maps, but you're welcome to edit the search_app.py and create your own maps.

* Uniform Cost Search

Depth-first search cannot find optimal paths, unless it gets lucky. To address this issue, the uniform_cost_search function in `search.py` implements the uniform cost search algorithm. Again, write a graph search algorithm that allows a state to be expanded only once. Your algorithm should return the optimal path from start to goal.

Hint: As all action costs in the grid world are unit costs, uniform cost search behaves like breadth-first search with both methods returning optimal solutions (breaking ties in the priority queue may result in visually different minimal cost routes). You can verify this by implementing BFS by simply changing the data structure used in your DFS implementation to expand the open list nodes in a FIFO manner. Of course, when the step costs vary, BFS cannot guarantee optimal paths. You can change the cost function in search_app.py to verify this.  

* A* search

The astar_search function in search.py to implement A* graph search. As A* needs a heuristic function to work, please modify the empty heuristic function accordingly to return the Manhattan distance between a given state and the goal state.

You should see that in general A* finds the optimal solution (slightly) faster than UCS. Note once more that the way you break ties in the priority queues may lead to different minimal cost paths for the two methods.

Hint: By setting the heuristic distance to 0, the A* search effectively behaves like uniform cost search. By comparing the two search algorithms, you should be able to see the advantages of heuristics search on certain scenarios.

----

# Important notes: 

All of your search functions should return a list of valid actions from the start to the goal, and the list of the expanded nodes, i.e. the closed set. See the code for more details.

You can find implementations of Queue, Stack (LIFO Queue), and PriorityQueue data structures in `search_app.py`. 

The code automatically visualizes the states in the open and closed list to help you debug your search algorithms. States on the open list are colored `white`, and expanded nodes are colored `red`.

---

# TO DO:
* Lines 244-247: You do not need a separate list to hold the children. You can generate them on the fly for a in ACTIONS: child = (currentNode[0] + a[0], currentNode[1] + a[1]). * Similarly, the actions if statement (line 255-262) are an overkill. You can simply do actions[child[0]][child[1]] = a, as you do for the parents in line 277. 
* Line 372: You do not have to call remove. Whenever you call put, it first checks if a node is on the priority queue, and if it is, it just updates its value.
