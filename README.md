# SMURF

Source code to accompany: End-to-end learning of multiple sequence alignments with differentiable Smith-Waterman
See: https://www.biorxiv.org/content/10.1101/2021.10.23.465204v1

[1] Petti S, Bhattacharya N, Rao R, Dauparas J, Thomas N, Zhou J, Rush AM, Koo PK, Ovchinnikov S. End-to-end learning of multiple sequence alignments with differentiable Smith-Waterman. BioRxiv. 2021 Oct 24.

### Source code for SSW and SMURF:

- laxy.py: Wrapper around JAX for basic neural networks, see https://github.com/sokrypton/laxy.

- sw_functions.py: Differentiable JAX implementations of smooth Smith Waterman and Needleman Wunsch. Features affine gap and temperature parameters. 

- network_functions.py: SMURF pipeline including the BasicAlign and TrainMRF modules.


### examples/SSW_examples: example usage and speed tests for SSW (Smooth Smith-Waterman):

- ssw_examples.ipynb: Tutorial on how to use our smooth Smith Waterman implementation.

- nw_speedtest.ipynb: Runtime comparison of our vectorized code to a naive implementation and to the "deepBLAST" implementation given in [Morton et al. 2020]. For implementations of both local (sw) and global (nw) alignment algorithms.

- sw_in_tensorflow_pytorch.ipynb: Implementations in TensorFlow and PyTorch

### examples/SMURF_examples: example usage of SMURF on RNA and protein and creation of associated figures

- run_smurf.py: Code that executes SMURF and MLM-GREMLIN. Selected hyperparameters described in the comments. Outputs a single file containing the contact prediction AUCs for the families.

- run_smurf_w_contacts_aln.py: Code that executes SMURF and MLM-GREMLIN. Selected hyperparameters described in the comments. Outputs a one file per family that contains the predicted contacts, contact prediction AUC, and learned alignment (for SMURF only).

- ablation_test.py: Code that executes ablations described in [1].

- data_description.txt: Description of data used in [1].

- make_SMURF_figures_protein.ipynb and make_SMURF_figures_RNA.ipynb: Code to generate figures in [1].


### examples/LAM_AF_examples: example usage of LAM with AF and creation of associated figures

- CASP_examples: folder containing MMSeqs2 generated alignments and true structures for the examples analyzed in [1].

- learned_alns: folder containing alignments learned by LAM for each family; generated via save_and_view_msas.ipynb.

- af_msa_backprop.ipynb: Backprop through AlphaFold to "learn" an MSA from a collection of unaligned sequences that maximizes the confidence metric (and hopefully returns a more accurate structure). Illustrates trajectories.

- af_opt_and_save_v2.py: Same pipeline as above, executes choice of random seed, learning rate, cooling, and E-value restrictions used in [1]

- make_AF_figures.ipynb: Plots figures that show pLDDT and RMSD of best points in each trajectory for each family

- make_pairwise_aln_figures.ipynb: Code to generate all figures in [1] relating to the pairwise alignments of learned MSAs (for both SMURF and AF)

- save_and_view_msas.ipynb: Code to construct MSAs from the saved parameters of an LAM. Results stored in learned_alns folder. Also displays the alignment shown in [1].

- sensitivity_of_AF_preds.ipynb: Code that evaluates the sensitivity of AF predictions to (a) the random mask and (b) the removal of sets of sequences. Generates related figures in [1].
