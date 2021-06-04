### Supplementary code for "Finding Hamiltonian cycles with Graph neural networks"

Clone and run main.py to get the evaluation of "HamS" and "HamR" models presented in the paper.


### Requirements:
* [Python](https://www.python.org/) 3.8 or later
* [PyTorch](https://pytorch.org/) 1.8 or later
* [PyTorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) 1.6.3 or later
* [Concorde TSP solver](http://www.math.uwaterloo.ca/tsp/concorde.html)

### Usage
* Running main.py loads HamS and HamR models and evaluates them on graphs stored in DATA. This is just a small sample of graphs, not the evaluation data used in the paper.
* Running visualizations.py will generate figures from the paper for two examples.


#### Exact solvers
Finding exact solutions requires that "Concorde TSP" is installed on the system.
In that case config.cfg has to be adjusted to point to "concorde" executable.

It contains the following fields:
* CONCORDE_SCRIPT_PATH - path to "concorde" executable
* CONCORDE_WORK_DIR - temporary directory (preferably empty) where intermediate inputs and outputs from concorde are created (it is cleaned automatically) 
* CONCORDE_INPUT_FILE - name of the input file which is created and used as input to concorde
