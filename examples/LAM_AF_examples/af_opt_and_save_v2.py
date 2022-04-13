
import os
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adam
import pickle

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('af_backprop')
sys.path.append('SMURF')
from utils import *

import laxy
import sw_functions as sw
import network_functions as nf

# import libraries
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.model import data, config, model
from alphafold.data import parsers
from alphafold.model import all_atom


os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/11.1.0-fasrc01"


def idx_below_thresh(t, evals):
    return [_ for _ in range(len(evals)) if evals[_]<t]

def restrict_idx(DOM, t):
    a3m_file = f"SMURF/examples/CASP_examples/{DOM}.mmseqs.id90cov75.a3m"
    evals = []
    fh = open(a3m_file, "r")
    for line in fh.readlines():
        if line[0]=='>':
            l = line.split()
            if len(l)<4 and len(evals)>0:
                raise ValueError("e value missing")
            if len(l)==1 and len(evals)==0:
                evals.append(0)
            else:
                evals.append(float(l[3]))
    return idx_below_thresh(t, evals)

# DOM restrict_to temp weight conv out_path

DOM = sys.argv[1] #"T1064-D1","T1038-D1","T1043-D1","T1039-D1"

e_val = sys.argv[2] 
if e_val == "None":
    restrict_to = None
else:
    restrict_to = restrict_idx(DOM, float(e_val))

    
print("restrict_to")
print(restrict_to)


temp = sys.argv[3]
if temp == "None":
    temps = None
elif temp == "Cool":
    temps = [3.0,2.5,2.0,1.5,1.0]
elif temp == "Short_Cool":
    temps = list(np.linspace(2.0,1.0,75)) + [1.0 for _ in range(25)]
elif temp == "Smooth_Cool":
    temps = list(np.linspace(3.0,1.0,75)) + [1.0 for _ in range(25)]
elif temp == "Gentle_Cool":
    temps = list(np.linspace(1.5,.75,100))
    
    
print("temps")
print(temps)

out_path = sys.argv[4]

weight = False

conv = True

if os.path.isdir(out_path):
    raise ValueError(f"results already written at {out_path}")
else:
    os.mkdir(out_path)
    
#params that will be iterated over 
#num_seeds = 30
num_seeds = 90
num_iters = 100
    
#params I that aren't being adjusted
mode = 'random'
adv_loss = None
confidence = True
supervised = False 
unsupervised = False
dropout = False
backprop_recycles = False


def get_feat(filename, alphabet="ARNDCQEGHILKMFPSTWYV"):
  '''
  Given A3M file (from hhblits)
  return MSA (aligned), MS (unaligned) and ALN (alignment)
  '''
  def parse_fasta(filename):
    '''function to parse fasta file'''    
    header, sequence = [],[]
    lines = open(filename, "r")
    for line in lines:
      line = line.rstrip()
      if len(line) == 0: pass
      else:
        if line[0] == ">":
          header.append(line[1:])
          sequence.append([])
        else:
          sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]
    return header, sequence

  names, seqs = parse_fasta(filename)  
  a2n = {a:n for n,a in enumerate(alphabet)}
  def get_seqref(x):
    n,seq,ref,aligned_seq = 0,[],[],[]
    for aa in list(x):
      if aa != "-":
        seq.append(a2n.get(aa.upper(),-1))
        if aa.islower(): ref.append(-1); n -= 1
        else: ref.append(n); aligned_seq.append(seq[-1])
      else: aligned_seq.append(-1)
      n += 1
    return seq, ref, aligned_seq
  
  # get the multiple sequence alignment
  max_len = 0
  ms, aln, msa = [],[],[]
  for seq in seqs:
    seq_,ref_,aligned_seq_ = get_seqref(seq)
    if len(seq_) > max_len: max_len = len(seq_)
    ms.append(seq_)
    msa.append(aligned_seq_)
    aln.append(ref_)
  
  return msa, ms, aln


# In[8]:


def prep_inputs(DOM, restrict_to = None):
  a3m_file = f"SMURF/examples/CASP_examples/{DOM}.mmseqs.id90cov75.a3m"
  _, ms, aln = get_feat(a3m_file)
  if restrict_to is not None:
    ms = [ms[_] for _ in restrict_to]
    aln = [aln[_] for _ in restrict_to]
  lens = np.asarray([len(m) for m in ms])
  ms = nf.one_hot(nf.pad_max(ms))
  aln = nf.one_hot(nf.pad_max(aln))
  N = len(ms)
  protein_obj = protein.from_pdb_string(pdb_to_string(f"SMURF/examples/CASP_examples/{DOM}.pdb"))
  batch = {'aatype': protein_obj.aatype,
          'all_atom_positions': protein_obj.atom_positions,
          'all_atom_mask': protein_obj.atom_mask}
  batch.update(all_atom.atom37_to_frames(**batch)) # for fape calculcation
  msa, mtx = parsers.parse_a3m(open(a3m_file,"r").read())
  if restrict_to is not None:
    msa = [msa[_] for _ in restrict_to]
    mtx = [mtx[_] for _ in restrict_to]
  feature_dict = {
      **pipeline.make_sequence_features(sequence=msa[0],description="none",num_res=len(msa[0])),
      **pipeline.make_msa_features(msas=[msa], deletion_matrices=[mtx])
  }
  feature_dict["residue_index"] = protein_obj.residue_index
  return {"N":N,"lens":lens,
          "ms":ms,"aln":aln,
          "feature_dict":feature_dict,
          "protein_obj":protein_obj, "batch":batch}


# In[9]:


def get_model_runner(num_seq, model_name="model_3_ptm", dropout=False, backprop_recycles=False):
  # setup which model params to use
  model_config = config.model_config(model_name)
  model_config.model.global_config.use_remat = True

  model_config.model.num_recycle = 3
  model_config.data.common.num_recycle = 3

  model_config.data.eval.max_msa_clusters = num_seq
  model_config.data.common.max_extra_msa = 1
  model_config.data.eval.masked_msa_replace_fraction = 0

  # backprop through recycles
  model_config.model.backprop_recycle = backprop_recycles
  model_config.model.embeddings_and_evoformer.backprop_dgram = backprop_recycles

  if not dropout:
    model_config = set_dropout(model_config,0)

  # setup model
  model_params = data.get_model_haiku_params(model_name=model_name, data_dir=".")
  model_runner = model.RunModel(model_config, model_params, is_training=True)
  return model_runner, model_params


# In[10]:


def get_grad_fn(model_runner, x_ref_len, confidence=True, supervised=False, unsupervised=False, batch=None, conv = True, weight = False):
  def mod(msa_params, key, inputs, model_params, msa_inputs, temp):  
    
    if conv:
        # get embedding per sequence
        emb = laxy.Conv1D(msa_params["emb"])(msa_inputs["x"])

        # get similarity matrix
        lengths = jnp.stack([msa_inputs["lengths"], jnp.broadcast_to(x_ref_len,msa_inputs["lengths"].shape)],-1)
        sm_mtx = emb @ emb[0,:x_ref_len].T
        sm_mask = jnp.broadcast_to(msa_inputs["x"].sum(-1,keepdims=True), sm_mtx.shape)
        sm_mtx = nf.norm_row_col(sm_mtx, sm_mask, norm_mode="fast")

        # get alignment
        aln = sw.sw()(sm_mtx, lengths, msa_params["gap"], temp)

        # get msa
        x_msa = jnp.einsum("...ia,...ij->...ja", msa_inputs["x"], aln)
        x_msa = x_msa.at[0,:,:].set(msa_inputs["x"][0,:x_ref_len,:])

        # add gap character
        x_gap = jax.nn.relu(1 - x_msa.sum(-1,keepdims=True))
        x_msa_gap = jnp.concatenate([x_msa,jnp.zeros_like(x_gap),x_gap],-1)

        # update msa
        inputs_mod = inputs
        inputs_mod["msa_feat"] = jnp.zeros_like(inputs["msa_feat"]).at[...,0:22].set(x_msa_gap).at[...,25:47].set(x_msa_gap)
        
        seq = x_msa[0]
    else:
        x_msa_gap = None 
        seq = None
        aln = None
        inputs_mod = inputs
    
    if weight:
        # reweight MSA
        inputs_mod["msa_feat"] = jnp.tanh(msa_params["weights"]).at[0].set(1)[None,..., None, None] * inputs_mod["msa_feat"]
        
    # get alphafold outputs
    outputs = model_runner.apply(model_params, key, inputs_mod)

    #################
    # compute loss
    #################
    # distance to correct solution
    rmsd_loss = jnp_rmsd(INPUTS["protein_obj"].atom_positions[:,1,:],
                         outputs["structure_module"]["final_atom_positions"][:,1,:])

    loss = 0
    losses = {"rmsd":rmsd_loss}
    if supervised:
      dgram_loss = get_dgram_loss(batch, outputs, model_config=model_runner.config)
      fape_loss = get_fape_loss(batch, outputs, model_config=model_runner.config)
      loss += dgram_loss + fape_loss
      losses.update({"dgram":dgram_loss, "fape":fape_loss})

    if unsupervised:
      x_msa_pred_logits = outputs["masked_msa"]["logits"]
      x_ms_pred_logits = jnp.einsum("...ja,...ij->...ia", x_msa_pred_logits, aln)
      x_ms_pred_log_softmax = jax.nn.log_softmax(x_ms_pred_logits[...,:22])[...,:20]
      cce_loss = -(msa_inputs["x"] * x_ms_pred_log_softmax).sum() / msa_inputs["x"].sum()
      loss += cce_loss
      losses.update({"cce":cce_loss})
      
    if confidence:
      pae_loss = jax.nn.softmax(outputs["predicted_aligned_error"]["logits"])
      pae_loss = (pae_loss * jnp.arange(pae_loss.shape[-1])).sum(-1).mean()
      plddt_loss = jax.nn.softmax(outputs["predicted_lddt"]["logits"])
      plddt_loss = (plddt_loss * jnp.arange(plddt_loss.shape[-1])[::-1]).sum(-1).mean()
      loss += pae_loss + plddt_loss
      losses.update({"pae":pae_loss, "plddt":plddt_loss})
      
    outs = {"final_atom_positions":outputs["structure_module"]["final_atom_positions"],
            "final_atom_mask":outputs["structure_module"]["final_atom_mask"]}

    return loss, ({"plddt": get_plddt(outputs),
                   "losses":losses, "outputs":outs,                   
                   "msa":x_msa_gap, "seq":seq, "aln":aln})
 
  
  return mod, jax.value_and_grad(mod, has_aux=True, argnums=0)




# prep inputs
INPUTS = prep_inputs(DOM, restrict_to)
msa_inputs = {"x":INPUTS["ms"], "aln":INPUTS["aln"], "lengths":INPUTS["lens"]}

# setup AlphaFold model
model_runner, model_params = get_model_runner(INPUTS["N"], dropout=dropout,backprop_recycles=backprop_recycles)
inputs = model_runner.process_features(INPUTS["feature_dict"], random_seed=0)
loss_fn, grad_fn = get_grad_fn(model_runner, x_ref_len=INPUTS["lens"][0],
                             confidence=confidence, unsupervised=unsupervised, supervised=supervised,
                             batch=INPUTS["batch"], conv = conv, weight = weight) 

# loss_fn returns loss, ({"plddt": get_plddt(outputs), "losses":losses, "outputs":outs,"msa":x_msa_gap, "seq":x_msa[0]})
# grad_fn returns pair, first is output of loss function
grad_fn = jax.jit(grad_fn)

LOSSES = []
# testing 6 MSA pretraining methods
# random = starting with random convolutions
# ba = pretrain convolutions using basic-align
# ba_sup = pretrain convolutions using basic-align to recreate the initial MSA
# smurf = pretrain convolutions using smurf
# ba_smurf = pretrain convolutions using smurf+basic-align
# ba_sup_smurf = pretrain convolutions using smurf+basic-align
suffix = f'{adv_loss}.{mode}'
if supervised: suffix += "_supervised"
if unsupervised: suffix += "_unsupervised"
if backprop_recycles: suffix += "_backprop-recycles"
npy_file = f"{out_path}/{DOM}.traj.{suffix}.npy"

if not os.path.isfile(npy_file):
  BEST_PLDDT = 0
  losses = []
  for seed in range(num_seeds):
 # for seed in range(num_seeds): # try different seeds and learning rates
  #  lr = [1e-2][seed%3]
    #lr = [1e-2,1e-3,1e-4][seed%3]
    lr = [1e-2,1e-3,1e-4][seed%3]
    print(f"seed {seed}, lr {lr}")
    init_fun, update_fun, get_params = adam(step_size=lr) 
    def step(i, state, key, inputs, model_params, msa_inputs, temp):
      (loss, outs), grad = grad_fn(get_params(state), key, inputs, model_params, msa_inputs, temp)
      state = update_fun(i, grad, state)
      return state, (loss,outs)

    ######################################################
    key = jax.random.PRNGKey(seed)
    ###################################
    # pretrain w/ BasicAlign
    ###################################
    msa_inputs = {"x":INPUTS["ms"], "aln":INPUTS["aln"], "lengths":INPUTS["lens"]}
    ns = INPUTS["ms"].shape[0]
    if mode == "random":
      if weight and conv:
          state = init_fun({"emb":laxy.Conv1D()(20,512,18,key=key),
                        "gap":jnp.full([],-3.0,dtype=jnp.float32), "weights": jnp.ones(ns, dtype = jnp.float32)})
      elif weight:
          state = init_fun({"weights": jnp.ones(ns, dtype = jnp.float32)})
      elif conv:
          state = init_fun({"emb":laxy.Conv1D()(20,512,18,key=key),
                        "gap":jnp.full([],-3.0,dtype=jnp.float32)})

    if "ba" in mode:
      if "sup" in mode:
        msa_model = nf.BasicAlign(X=INPUTS["ms"],lengths=INPUTS["lens"],
                                  sw_open=None, sw_gap=-3.0, sw_learn_gap=True,
                                  seed=seed, batch_size=INPUTS["N"],
                                  supervise=True, nat_aln=INPUTS["aln"])
      else:
        msa_model = nf.BasicAlign(X=INPUTS["ms"],lengths=INPUTS["lens"],
                                  sw_open=None, sw_gap=-3.0, sw_learn_gap=True,
                                  seed=seed, batch_size=INPUTS["N"])
    if "smurf" in mode:
      smurf_model = nf.MRF(X=INPUTS["ms"], lengths=INPUTS["lens"], batch_size=INPUTS["N"],
                          sw_open=None, sw_gap=-3.0, sw_learn_gap=True, seed=seed)
      if "ba" in mode:
        _ = msa_model.fit(2000, verbose=False)
        msa_params = msa_model.opt.get_params()

        smurf_params = smrf_model.opt.get_params()
        smurf_params.update({k:msa_params[k] for k in ["emb","gap","open"]})
        smurf_model.opt.set_params(smurf_params)

      _ = smurf_model.fit(2000, verbose=False)
      smurf_params = smurf_model.opt.get_params()
      state = init_fun({"emb":smurf_params["emb"],"gap":smurf_params["gap"]})    

    elif "ba" in mode:
      _ = msa_model.fit(4000, verbose=False)
      msa_params = msa_model.opt.get_params()
      state = init_fun({"emb":msa_params["emb"],"gap":msa_params["gap"]})    

    # optimize!
    losses.append([])
    for i in range(num_iters):
      if temps is not None:
        temp = temps[int(i/(num_iters/len(temps)))]
      else:
        temp = 1.0
      print(f"iter {i}, temp {temp}")
      key,subkey = jax.random.split(key)
      state, (loss,outs) = step(i, state, subkey, inputs, model_params, msa_inputs, temp)
      plddt = outs["plddt"].mean()
      # save results
      losses[-1].append([loss,plddt])
      losses[-1][-1].append(outs["losses"]["rmsd"])
      if unsupervised: losses[-1][-1].append(outs["losses"]["cce"])

      print(DOM, seed, i, *losses[-1][-1])
    
      # save outputs
     # if conv: save_pdb(outs,f"{out_path}/{DOM}.{suffix}.{seed}.{lr}.{i}.pred.pdb")
     # pickle.dump(outs, open(f"{out_path}/{DOM}.{suffix}.{seed}.{lr}.{i}.out_dict","wb"))
     # pickle.dump({"lr":lr, "seed": seed, "DOM" : DOM, "restrict_to":restrict_to, "temps": temps, "weight":weight, "conv":conv, "mode":mode}, open(f"{out_path}/{DOM}.{suffix}.{seed}.{lr}.{i}.p_dict.best","wb"))
     # if weight:
      #    pickle.dump(get_params(state)["weights"], open(f"{out_path}/{DOM}.{suffix}.{seed}.{lr}.{i}.weight","wb"))
      
      # save best with special note "best"
      if plddt > BEST_PLDDT:
        BEST_PLDDT = plddt
        if conv: save_pdb(outs,f"{out_path}/{DOM}.{suffix}.pred.pdb.best")
        pickle.dump(outs, open(f"{out_path}/{DOM}.{suffix}.out_dict.best","wb"))
        pickle.dump({"lr":lr, "seed": seed, "DOM" : DOM, "restrict_to":restrict_to, "temps": temps, "weight":weight, "conv":conv, "mode":mode}, open(f"{out_path}/{DOM}.{suffix}.p_dict.best","wb"))
        if weight:
            pickle.dump(get_params(state)["weights"], open(f"{out_path}/{DOM}.{suffix}.weight.best","wb"))
  np.save(npy_file, losses)
else:
  losses = np.load(npy_file)

LOSSES.append(losses) 






