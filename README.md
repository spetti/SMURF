# SMURF

### Source code for SSW and SMURF:

laxy.py: Wrapper around JAX for basic neural networks, see https://github.com/sokrypton/laxy

sw_functions.py: Differentiable JAX implementations of smooth Smith Waterman and Needleman Wunsch. Features affine gap and temperature parameters. 

network_functions.py: SMURF pipeline including the BasicAlign and TrainMRF modules 



### Examples of SMURF and SSW:

ssw_examples.ipynb: Tutorial on how to use our smooth Smith Waterman implementation.

run_smurf.py: Code that executes SMURF and MLM-GREMLIN. Selected hyperparameters described in the comments. Outputs a single file containing the contact prediction AUCs for the families.

run_smurf_w_contacts_aln.py: Code that executes SMURF and MLM-GREMLIN. Selected hyperparameters described in the comments. Outputs a one file per family that contains the predicted contacts, contact prediction AUC, and learned alignment (for SMURF only).

ablation_test.py: Code that excu

nw_speedtest.ipynb: Runtime comparison of our vectorized code to a naive implementation and to the "deepBLAST" implementation given in [Morton et al. 2020]



### Data and Figures:

data_description.txt: Description of data used in [citation coming soon].

make_SMURF_figures.ipynb: Code to generate figures in [citation coming soon].

