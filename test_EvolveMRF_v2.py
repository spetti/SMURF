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
batch_size = sys.argv[3] #placeholder now
iters = sys.argv[4] # S (2000 BasicAlign + 1000 MRF) or MB (extra 1000 iters on BasicAlign) or MM (extra 1000 iters on MRF)
gradual = sys.argv[5] # yes or no
msa_frac = float(sys.argv[6]) # .3, .5, .7
msa_memory = float(sys.argv[7]) # .9 or .95
num_fams = 190

mode = "EvolveMRF"
gap = - 3



try:
    aucs, losses, fams = pickle.load(open(f"results/{date}_{num_fams}_{mode}_{lr}_{batch_size}_{iters}_{gradual}_{msa_frac}_{msa_memory}","rb"))
    print("starting from existing file")
except:
    print("starting from scratch")
    aucs = []
    losses = []
    fams = []
    
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
    
if iters == "S":
    basic_steps = [int(2000*_) for _ in basic_steps]
    MRF_steps = [int(1000*_) for _ in MRF_steps]
if iters == "MB":
    basic_steps = [int(2000*_) for _ in basic_steps]
    MRF_steps = [int(2000*_) for _ in MRF_steps]
if iters == "MM":
    basic_steps = [int(3000*_) for _ in basic_steps]
    MRF_steps = [int(1000*_) for _ in MRF_steps]

data = np.load("data_unalign.npz", allow_pickle=True)
for n,x in enumerate(data.keys()):
    
    #if x not in special_fams: continue
    if x in fams: continue
    a = data[x].item()

    # prep data
    seqs = nf.sub_sample(a["ms"])
    lens = np.array([len(seq) for seq in seqs])
    ms = nf.one_hot(nf.pad_max(seqs))
    aln = nf.one_hot(nf.pad_max(nf.sub_sample(a["aln"])))

    if "model" in globals():
        del model
        nf.clear_mem()
    #if True:
    for batch_size in [265,128,64]:
        try:
            if mode != "Gremlin":
                model = nf.BasicAlign(X=ms, lengths=lens, batch_size=batch_size, filters=512, win=18, 
                           sw_unroll=4, sw_temp=1.0, sw_learn_temp=False,
                           sw_open=None, sw_gap=gap, sw_learn_gap=True,
                           sw_restrict=False,
                           seed=None, lr=lr, norm_mode="fast", 
                           w_scale=0.1)

                model.fit(basic_steps[0], verbose=False)
    #             if mode in ["DoubleAlign", "Gradual"]:
    #                 model.reset_model_and_opt({"double_frac":msa_frac, "double_aln": True, "pid_thresh": basic_pid[1]})
    #                 loss+=model.fit(basic_steps[1], verbose=False)

    #                 model.reset_model_and_opt({"double_frac":msa_frac, "double_aln": True, "pid_thresh":basic_pid[2]})
    #                 loss+=model.fit(basic_steps[2], verbose=False)

    #                 model.reset_model_and_opt({"double_frac":msa_frac, "double_aln": True, "pid_thresh":basic_pid[3]})
    #                 loss+=model.fit(basic_steps[3], verbose=False)

                msa_params = model.opt.get_params()

                print("MRF")

                auc = []
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
                    if p=="gap": print(p, msa_params[p])
                    mrf_params[p] = msa_params[p]
                model.opt.set_params(mrf_params)
                model.fit(MRF_steps[0], verbose=False)
                auc.append(model.get_auc())

                if mode in ["EvolveMRF", "Gradual"]:
                    model.reset_model_and_opt({"msa_memory": msa_memory, "align_to_msa_frac": msa_frac, "pid_thresh":MRF_pid[1]})
                    model.fit(MRF_steps[1], verbose=False)
                    auc.append(model.get_auc())


                    model.reset_model_and_opt({"msa_memory": msa_memory, "align_to_msa_frac": msa_frac, "pid_thresh":MRF_pid[2]})
                    model.fit(MRF_steps[2], verbose=False)
                    auc.append(model.get_auc())


                    model.reset_model_and_opt({"msa_memory": msa_memory, "align_to_msa_frac": msa_frac, "pid_thresh":MRF_pid[3]})
                    model.fit(MRF_steps[3], verbose=False)
                    auc.append(model.get_auc())


                print("X", x, model.get_auc(), batch_size)
                break

    #         if mode == "Gremlin":
    #             model = nf.MRF(X=ms, lengths=lens, ss_hide=0.15, batch_size=batch_size, 
    #                filters=512, win=18, lam=0.01,
    #                sw_unroll=4, sw_temp=1.0, sw_learn_temp=False,
    #                sw_open=None, sw_gap=None, sw_learn_gap=False,
    #                nat_contacts=a["true"], nat_contacts_mask=a["mask"],
    #                nat_aln=jnp.array(aln), use_nat_aln=True, add_aln_loss=False, aln_lam=1.0,
    #                seed=None, lr=lr, norm_mode="fast",
    #                learn_bias=True, w_scale=0.1, 
    #                msa_memory = init_msa_memory, align_to_msa_frac = init_align_to_msa_frac, pid_thresh = MRF_pid[0])
    #             model.fit(MRF_steps[0], verbose=True)
    #             auc = [model.get_auc()]
    #             loss = None
        #if False:        
        except:
            print(f"FAIL ON {x} with batch size {batch_size}")
            auc = None
            loss = None
        
    aucs.append(auc)
    fams.append(x)
    losses.append(None)
    batch_size = "highest"
    pickle.dump((aucs, losses, fams),open(f"results/{date}_{num_fams}_{mode}_{lr}_{batch_size}_{iters}_{gradual}_{msa_frac}_{msa_memory}","wb"))

        
    if n == num_fams -1 : break


