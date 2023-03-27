### Supplementary code for "Finding Hamiltonian cycles with Graph neural networks"

Clone and run train.py to get the evaluation of "HamS" and "HamR" models presented in the paper. Running generate_figures.py will generate figures of Erdos-Renyi graphs similar to those presented in the paper.


### Requirements:
* [Python](https://www.python.org/) 3.8 or later
* [PyTorch](https://pytorch.org/) 1.8 or later
* [PyTorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) 1.6.3 or later
* [Networkit](https://networkit.github.io/)
* [Concorde TSP solver](http://www.math.uwaterloo.ca/tsp/concorde.html) (optional)

### Paper
* the accompanying paper will be available soon ...

### Usage
* Train the model by running
```
python terminal_scripts/train_main_model.py
```
This should log a wandb run and produce a checkpoint in `MODEL_CHECKPOINTS` directory. Training data is generated on the fly so there is no need to aquire any data for training.
* Generate data for various experiments by running
```
python terminal_scripts/generate_data.py
```
After completion, this should create several dataset in the `./DATA` folder.
* The model can be tested on generated data by running
```
python terminal_scripts/test_main_model.py
```
It prints the output in somewhat raw format, the percentage of HCP solved can be seen in `perc_hamilton_found` column. A nicer output format will be added soon...
* To generate all the figures presented in the paper, run
```
python terminal_scripts/generate_figures.py
```
**Note, this will take a very long time since several different models need to be trained.**    
* A list of model that have been tried out during experimentation can be found in `hamgnn/models_list.py`. All models listed there should theoretically be runnable with:
```
python terminal_scripts/train_model_variation.py <experiment_name>
```
where <experiment_name> corresponds to the name of the python variable to which the experiment has been assigned inside `hamgnn/models_list.py` file.

### Exact solvers
Finding exact solutions requires that "Concorde TSP" is installed on the system.
In that case config.cfg has to be adjusted to point to "concorde" executable.

It contains the following fields (only the first field needs to be changed):
* CONCORDE_SCRIPT_PATH - path to "concorde" executable
* CONCORDE_WORK_DIR - temporary directory (preferably empty) where intermediate inputs and outputs from concorde are created (it is cleaned automatically) 
* CONCORDE_INPUT_FILE - name of the input file which is created and used as input to concorde

##Notes:
* The code in `hamgnn/legacy/``` is depricated and will be removed eventually