# Fair Hypernetworks (FHN)
Code for Fair Hypernetworks. Paper in progress.

The code is broken apart into five main files: 

1. fhn.py
2. fflvfedavg.py
3. decentralized.py
4. fedfb.py
5. fairfed.py

Each of these files calls to the dataset.py, node.py, and utils.py files which contain the code to parition the datasets (adult and compas), the code to generate the clients, and the code for helper functions, respectively. 

To configure/run the experiments performed in the paper, open the wanted experiment file and alter the parameters (e.g., dataset name, fairness type, so on) in the main() function at the bottom of the file. Then, on the command line run: 

    python file_name.py

The code relies on the following dependencies: 
 * Python = 3.8.10
 * Torch = 1.10.1
 * Numpy = 1.20.3
 * Pandas = 1.3.4
 * Tqdm = 4.62.3
 * Sci-kit learn = 0.24.2