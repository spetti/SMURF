import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import laxy
import pickle
import sys
import os
import sw_functions as sw
import network_functions as nf

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/10.1.243-fasrc01/"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

date = sys.argv[1]
lr = float(sys.argv[2]) # .05, .10
batch_size = sys.argv[3] # highest or a batch size
iters = sys.argv[4] # S (2000 BasicAlign + 1000 MRF) or MB (extra 1000 iters on BasicAlign) or MM (extra 1000 iters on MRF)
gradual = sys.argv[5] # yes or no
msa_frac = float(sys.argv[6]) # .3, .5, .7
msa_memory = float(sys.argv[7]) # .9 or .95
mode = sys.argv[8]
n_min = int(sys.argv[9])
n_max = int(sys.argv[10])
out_dir = sys.argv[11]


o_batch_size = batch_size
if batch_size == "highest":
    batch_sizes = [256, 128, 64]
else:
    batch_sizes = [int(batch_size)]
gap = - 3


store_convolution = True

basic_pid = [None, None, None, None]
basic_steps = [1, 0, 0, 0]
if gradual == "yes":
    MRF_pid = [1.00, .60, .40, .20]
    init_align_to_msa_frac = 0.0
    init_msa_memory = False

elif gradual == "no":
    MRF_pid = [0, 0, 0, 0]
    init_align_to_msa_frac = msa_frac
    init_msa_memory = msa_memory

MRF_steps = [1/4, 1/4, 1/4, 1/4]
basic_steps = [1,0,0,0]
    

if iters == "MM":
    basic_steps = [int(3000*_) for _ in basic_steps]
    MRF_steps = [int(1000*_) for _ in MRF_steps]
if iters == "G4":
    basic_steps = [0,0,0,0]
    MRF_steps = [4000, 0, 0, 0]

    
data = np.load("data_unalign.npz", allow_pickle=True)
verbose = False

for n,x in enumerate(data.keys()):
    if n >= n_max or n < n_min : continue
    if verbose: print(f"family {x}")
    a = data[x].item()

    # prep data
    seqs = nf.sub_sample(a["ms"])
    lens = np.array([len(seq) for seq in seqs])
    ms = nf.one_hot(nf.pad_max(seqs))
    aln = nf.one_hot(nf.pad_max(nf.sub_sample(a["aln"])))

    if "model" in globals():
        del model
        nf.clear_mem()
    for batch_size in batch_sizes:
        try:
            if verbose: print(f"attempting batch size {batch_size}")
            if mode != "Gremlin":
                model = nf.BasicAlign(X=ms, lengths=lens, batch_size=batch_size, filters=512, win=18, 
                           sw_unroll=4, sw_temp=1.0, sw_learn_temp=False,
                           sw_open=None, sw_gap=gap, sw_learn_gap=True,
                           sw_restrict=False,
                           seed=None, lr=lr, norm_mode="fast", 
                           w_scale=0.1)
                if verbose: print("BasicAlign")
                model.fit(basic_steps[0], verbose=verbose)
                msa_params = model.opt.get_params()

                if verbose: print("MRF")

                model = nf.MRF(X=ms, lengths=lens, ss_hide=0.15, batch_size=batch_size, 
                           filters=512, win=18, lam=0.01,
                           sw_unroll=4, sw_temp=1.0, sw_learn_temp=False,
                           sw_open=None, sw_gap=None, sw_learn_gap=False,
                           nat_contacts=a["true"],
                                nat_contacts_mask=a["mask"],
                           nat_aln=None, use_nat_aln=False, add_aln_loss=False, aln_lam=1.0,
                           seed=None, lr=lr, norm_mode="fast",
                           learn_bias=True, w_scale=0.1, 
                           msa_memory = init_msa_memory, align_to_msa_frac = init_align_to_msa_frac, pid_thresh = MRF_pid[0])
                mrf_params = model.opt.get_params()
                for p in ["emb","gap","open"]:
                    if p=="gap" and verbose: print(p, msa_params[p])
                    mrf_params[p] = msa_params[p]
                model.opt.set_params(mrf_params)
                model.fit(MRF_steps[0], verbose=verbose)

                if mode == "EvolveMRF":
                    if verbose: print(f"reset with pid thresh: {MRF_pid[1]}")
                    model.reset_model_and_opt({"msa_memory": msa_memory, "align_to_msa_frac": msa_frac, "pid_thresh":MRF_pid[1]})
                    model.fit(MRF_steps[1], verbose=verbose)
                    
                    if verbose: print(f"reset with pid thresh: {MRF_pid[2]}")
                    model.reset_model_and_opt({"msa_memory": msa_memory, "align_to_msa_frac": msa_frac, "pid_thresh":MRF_pid[2]})
                    model.fit(MRF_steps[2], verbose=verbose)

                    if verbose: print(f"reset with pid thresh: {MRF_pid[3]}")
                    model.reset_model_and_opt({"msa_memory": msa_memory, "align_to_msa_frac": msa_frac, "pid_thresh":MRF_pid[3]})
                    model.fit(MRF_steps[3], verbose=verbose)


                print("X", mode, x, model.get_auc(), batch_size)
                
                if store_convolution:
                    c_sw_param = {}
                    mrf_params = model.opt.get_params()
                    for p in ["emb","gap","open"]:
                        c_sw_param[p]=mrf_params[p]
                    

            if mode == "Gremlin":
                if verbose: print("MRF")
                model = nf.MRF(X=ms, lengths=lens, ss_hide=0.15, batch_size=batch_size, 
                   filters=512, win=18, lam=0.01,
                   sw_unroll=4, sw_temp=1.0, sw_learn_temp=False,
                   sw_open=None, sw_gap=None, sw_learn_gap=False,
                   nat_contacts=a["true"], nat_contacts_mask=a["mask"],
                   nat_aln=jnp.array(aln), use_nat_aln=True, add_aln_loss=False, aln_lam=1.0,
                   seed=None, lr=lr, norm_mode="fast",
                   learn_bias=True, w_scale=0.1)
                model.fit(MRF_steps[0], verbose=verbose)
                
            
            auc = model.get_auc()
            p_contacts = model.get_contacts()
            
      
            break
            
        except:
            print(f"FAIL ON {x} with batch size {batch_size}")
            auc = None
            p_contacts = None
        
    
    pickle.dump((auc, p_contacts, batch_size),open(f"{out_dir}/{date}_{x}_{mode}_{lr}_{o_batch_size}_{iters}_{gradual}_{msa_frac}_{msa_memory}","wb"))

    if store_convolution and mode!="Gremlin":
        pickle.dump(c_sw_param,open(f"{out_dir}/{date}_cparams_{x}_{mode}_{lr}_{o_batch_size}_{iters}_{gradual}_{msa_frac}_{msa_memory}","wb"))

