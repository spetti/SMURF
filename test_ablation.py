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
batch_size = sys.argv[2] # highest or a batch size
mode = sys.argv[3] # noevolve, nobasicalign

lr = .05
num_fams = 190
o_batch_size = batch_size
if batch_size == "highest":
    batch_sizes = [256, 128, 64]
else:
    batch_sizes = [int(batch_size)]
gap = - 3


try:
    aucs, losses, fams = pickle.load(open(f"results/{date}_{mode}_{lr}_{o_batch_size}","rb"))
    print("starting from existing file")
except:
    print("starting from scratch")
    aucs = []
    losses = []
    fams = []

pseudo = False
    
if mode == "noevolve":
    basic_steps = 3000
    MRF_steps = 1000
    
elif mode == "nobasicalign":
    MRF_steps = 4000

elif mode == "pseudo":
    basic_steps = 3000
    MRF_steps = 1000
    pseudo= True

data = np.load("data_unalign.npz", allow_pickle=True)
verbose = False
aucs = []
fams = []
for n,x in enumerate(data.keys()):
    if n <= num_fams -1 : continue
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
                if mode != "nobasicalign":
                    model = nf.BasicAlign(X=ms, lengths=lens, batch_size=batch_size, filters=512, win=18, 
                               sw_unroll=4, sw_temp=1.0, sw_learn_temp=False,
                               sw_open=None, sw_gap=gap, sw_learn_gap=True,
                               sw_restrict=False,
                               seed=None, lr=lr, norm_mode="fast", 
                               w_scale=0.1, pseudo=pseudo)
                    if verbose: print("BasicAlign")
                    model.fit(basic_steps, verbose=verbose)
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
                           learn_bias=True, w_scale=0.1, pseudo=pseudo)
                
                if mode != "nobasicalign":
                    mrf_params = model.opt.get_params()
                    for p in ["emb","gap","open"]:
                        if p=="gap" and verbose: print(p, msa_params[p])
                        mrf_params[p] = msa_params[p]
                    model.opt.set_params(mrf_params)
                model.fit(MRF_steps, verbose=verbose)

                print("X", mode, x, model.get_auc(), batch_size)
                

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
                model.fit(MRF_steps[0], verbose=True)
            
            auc = model.get_auc()
            break
        except:
            print(f"FAIL ON {x} with batch size {batch_size}")
            auc = None
        
    aucs.append(auc)
    fams.append(x)
        
    pickle.dump((aucs, [], fams),open(f"results/{date}_{mode}_{lr}_{o_batch_size}","wb"))

        
    #if n == num_fams -1 : break


