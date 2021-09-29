import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import laxy
import pickle
import sys
import os
import sw_functions as sw

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/10.1.243-fasrc01/"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# Written by Sergey Ovchinnikov and Sam Petti 
# Spring 2021

def sub_sample(x, samples=1024, seed=0):
  np.random.seed(seed)
  idx = np.arange(1,len(x))
  idx = np.random.choice(idx,samples-1,replace=False)
  idx = np.append(0,idx)
  return [x[i] for i in idx]

def pad_max(a, pad=-1):
  max_L = len(max(a,key = lambda x: len(x)))
  b = np.full([len(a),max_L], pad)
  for i,j in enumerate(a): b[i][0:len(j)] = j
  return b

def one_hot(x,cat=None):
  if cat is None: cat = np.max(x)+1
  oh = np.concatenate((np.eye(cat),np.zeros([1,cat])))
  return oh[x]

def con_auc(true, pred, mask=None, thresh=0.01):
  '''compute agreement between predicted and measured contact map'''
  if mask is not None:
    idx = mask.sum(-1) > 0
    true = true[idx,:][:,idx]
    pred = pred[idx,:][:,idx]
  eval_idx = np.triu_indices_from(true, 6)
  pred_, true_ = pred[eval_idx], true[eval_idx] 
  L = (np.linspace(0.1,1.0,10)*len(true)).astype("int")
  sort_idx = np.argsort(pred_)[::-1]
  acc = [(true_[sort_idx[:l]] > thresh).mean() for l in L]
  return np.mean(acc)

def clear_mem():
  backend = jax.lib.xla_bridge.get_backend()
  for buf in backend.live_buffers(): buf.delete()

def norm_row_col(z, z_mask, norm_mode):
  
  if norm_mode == "fast":
    z *= z_mask
    z -= (z.sum(1,keepdims=True)*z.sum(2,keepdims=True))/z.sum([1,2],keepdims=True)
    z *= z_mask
    z_sq = jnp.square(z)
    z /= jnp.sqrt((z_sq.sum(1,keepdims=True)*z_sq.sum(2,keepdims=True)) + 1e-8) / jnp.sqrt(z_sq.sum([1,2],keepdims=True) + 1e-8)

  if norm_mode == "slow":
    z_num = [None,z_mask.sum(1,keepdims=True),z_mask.sum(2,keepdims=True)]
    z *= z_mask
    for _ in range(2):
      for k in [1,2]:
        z -= z.sum(k,keepdims=True)/(z_num[k]+1e-8)
        z *= z_mask
        z /= jnp.sqrt(jnp.square(z).sum(k,keepdims=True)/(z_num[k]+1e-8) + 1e-8)

  if norm_mode == "simple":
    z *= z_mask
    z -= z.sum([1,2],keepdims=True)/z_mask.sum([1,2],keepdims=True)
    z *= z_mask
    z_sq = jnp.square(z)
    z /= jnp.sqrt(z_sq.sum([1,2],keepdims=True)/z_mask.sum([1,2],keepdims=True) + 1e-8)
  return z

def Conv1D_custom(params=None):
  '''convolution'''
  def init_params(in_dims, out_dims, win, key):
    return {"w":jnp.zeros((out_dims,in_dims,win)),
            "b":jnp.zeros(out_dims)}  
  def layer(x, use_bias=True, stride=1, padding="SAME", key=None, scale=0.1):
    w = params["w"]
    if key is not None:
      w += scale * jax.random.normal(key, shape=w.shape)
    x = x.transpose([0,2,1])
    y = jax.lax.conv(x,w,(stride,),padding=padding)
    y = y.transpose([0,2,1]) 
    if use_bias: y += params["b"]
    return y
  if params is None: return init_params
  else: return layer

##############################################################

class MRF:
    '''GRUMLIN implemented in jax'''
    def __init__(self, X, lengths=None, ss_hide=0.15, batch_size=128, 
               filters=512, win=18, lam=0.01,
               sw_unroll=4, sw_temp=1.0, sw_learn_temp=False,
               sw_open=None, sw_gap=None, sw_learn_gap=False,
               nat_contacts=None, nat_contacts_mask=None,
               nat_aln=None, use_nat_aln=False, add_aln_loss=False, aln_lam=1.0,
               seed=None, lr=0.1, norm_mode="fast",
               learn_bias=True, w_scale=0.1, 
               msa_memory = False, align_to_msa_frac = 0.0, pid_thresh = 1.0, pseudo = False):

        N,L,A = X.shape

        # inputs
        self.X = X
        self.lengths = X.sum([1,2]).astype(int) if lengths is None else lengths
        self.X_ref = self.X[:1]
        self.X_ref_len = self.lengths[0]

        self.nat_contacts = nat_contacts
        self.nat_contacts_mask = nat_contacts_mask
        self.nat_aln = nat_aln

        self.lr = lr*jnp.log(batch_size)/self.X_ref_len

        # seed for weight initialization and sub-sampling input
        self.key = laxy.KEY(seed)

        # params
        self.p = {"N":N, "L":L, "A":A, "batch_size":batch_size,
                  "sw_temp":sw_temp,"sw_learn_temp":sw_learn_temp,
                  "sw_unroll":sw_unroll,
                  "sw_open":sw_open,"sw_gap":sw_gap,"sw_learn_gap":sw_learn_gap,
                  "filters":filters, "win":win,
                  "x_ref_len":self.X_ref_len,
                  "ss_hide":ss_hide, "lam":lam*ss_hide*batch_size/N,
                  "use_nat_aln":use_nat_aln, "add_aln_loss":add_aln_loss, "aln_lam":aln_lam,
                  "norm_mode":norm_mode,
                  "learn_bias":learn_bias,"w_scale":w_scale, "msa_memory":msa_memory, 
                  "align_to_msa_frac":align_to_msa_frac, "pid_thresh":pid_thresh, "pseudo":pseudo}

        # initialize model
        self.init_params, self.model = self._get_model()

        self.model_aln = jax.jit(self._get_model(initialize_params=False, return_aln=True))
        self.opt = laxy.OPT(self.model, self.init_params, lr=self.lr)

  #####################################################################################################
  #####################################################################################################
    def _get_model(self, initialize_params=True, return_aln=False):
        p = self.p
        #######################
        # initialize params
        #######################
        if initialize_params:
            _params = {"mrf": laxy.MRF()(p["x_ref_len"], p["A"],
                                       use_bias=p["learn_bias"], key=self.key.get())}
            _params["emb"] = Conv1D_custom()(p["A"],p["filters"],p["win"],key=self.key.get())

            _params["open"] = p["sw_open"]
            _params["gap"] = p["sw_gap"]
            _params["temp"] = p["sw_temp"]
            _params["msa"] = self.X[0,:p["x_ref_len"],...]

        # self-supervision
        def self_sup(x, key=None):
            if p["ss_hide"] == 1 or key is None:
                return x,x
            else:
                tmp = jax.random.uniform(key,[p["batch_size"],p["L"],1])
                mask = (tmp > p["ss_hide"]).astype(x.dtype)
                return x*mask, x*(1-mask)

        # get alignment
        def get_aln(z, lengths, gap=None, open=None, temp=1.0, key=None): 
            # local-alignment (smith-waterman)
            if gap is None:
                aln_app = sw.sw_nogap(batch=True, unroll=p["sw_unroll"])
                aln = aln_app(z, lengths, temp)
            elif open is None:
                aln_app = sw.sw(batch=True, unroll=p["sw_unroll"])
                aln = aln_app(z, lengths, gap, temp)
            else:
                aln_app = sw.sw_affine(restrict_turns=True, batch=True, unroll=p["sw_unroll"])
                aln = aln_app(z, lengths, gap, open, temp)
            return aln

        #######################
        # setup the model
        #######################
        def _model(params, inputs):      
            # self-supervision (aka. masked-language-modeling)
            x_ms_in, x_ms_out = self_sup(inputs["x"], key=inputs["key"][0])

            if p["use_nat_aln"]:
                aln = p_aln = inputs["aln"]
                
            else:
            # concatentate reference, get positional embedding
                x_ms_in_ = jnp.concatenate([inputs["x_ref"], x_ms_in],0)
                emb = Conv1D_custom(params["emb"])(x_ms_in_,key=inputs["key"][1],scale=p["w_scale"])

                # get alignment to reference
                if p["align_to_msa_frac"]>0:
                    embedded_msa = Conv1D_custom(params["emb"])(params["msa"][None,...],key=inputs["key"][1],scale=p["w_scale"])
                    embedded_msa = embedded_msa[0,:p["x_ref_len"],...]
                    sm_mtx = emb[1:] @ ((1-self.p["align_to_msa_frac"]) * emb[0,:p["x_ref_len"]].T + self.p["align_to_msa_frac"] * embedded_msa.T)
                else:
                    sm_mtx = emb[1:] @ emb[0,:p["x_ref_len"]].T
                    
                # mask
                sm_mask = jnp.broadcast_to(inputs["x"].sum(-1,keepdims=True), sm_mtx.shape)
                lengths = jnp.stack([inputs["lengths"],
                                     jnp.broadcast_to(p["x_ref_len"],inputs["lengths"].shape)],-1)
                
                # normalize rows/cols (to remove edge effects due to 1D-convolution)
                sm_mtx = norm_row_col(sm_mtx, sm_mask, p["norm_mode"])
                
                
                if p["pseudo"]:
                    aln = jnp.sqrt(jax.nn.softmax(sm_mtx, axis=-1) * jax.nn.softmax(sm_mtx, axis=-2))
                else:
                    sm_open = params["open"] if p["sw_learn_gap"] else laxy.freeze(params["open"])
                    sm_gap = params["gap"] if p["sw_learn_gap"] else laxy.freeze(params["gap"])
                    sm_temp = params["temp"] if p["sw_learn_temp"] else laxy.freeze(params["temp"])
                    aln = get_aln(sm_mtx, lengths, gap=sm_gap, open=sm_open, temp=sm_temp, key=inputs["key"][1])
                    
                x_msa = jnp.einsum("nia,nij->nja", x_ms_in, aln)
                x_msa_bias = x_msa.mean(0)
  
                # update MSA 
                if self.p["msa_memory"] != False:
                    if p["pid_thresh"]<=1.0 and p["pid_thresh"]>0:
                        pid  = jnp.einsum('nla,la->n', x_msa, x_msa[0,...])/ x_msa.shape[1]
                        x_msa_restricted = jnp.einsum('nia,n->nia',x_msa, (pid > p["pid_thresh"]))
                        num_surviving_seqs = (pid > p["pid_thresh"]).sum() + 1
                        x_msa_bias_restricted = (self.X[0,:p["x_ref_len"],...] + x_msa_restricted.sum(axis = 0))/num_surviving_seqs
                    else:
                        x_msa_bias_restricted = x_msa_bias
                    params["msa"] = self.p["msa_memory"] * params["msa"] + (1-self.p["msa_memory"])* x_msa_bias_restricted[:p["x_ref_len"],...]

                laxy.freeze(params["msa"])

            if return_aln:
                return aln, sm_mtx

            # align, gremlin, unalign
            x_msa = jnp.einsum("nia,nij->nja", x_ms_in, aln)
            x_msa_pred, w = laxy.MRF(params["mrf"])(x_msa, return_w=True)
            if p["learn_bias"] == False:
                x_msa_pred += jnp.log(x_msa.sum(0) + 0.01 * p["batch_size"])
            x_ms_pred_logits = jnp.einsum("nja,nij->nia", x_msa_pred, aln)

            x_ms_pred = jax.nn.softmax(x_ms_pred_logits, -1)

            # regularization
            l2_loss = 0.5*(p["L"]-1)*(p["A"]-1)*jnp.square(w).sum() 
            if p["learn_bias"]:
                l2_loss += jnp.square(params["mrf"]["b"]).sum()

            # compute loss (pseudo-likelihood)
            cce_loss = -(x_ms_out * jnp.log(x_ms_pred + 1e-8)).sum()
            loss = cce_loss + p["lam"] * l2_loss

            if p["add_aln_loss"]:
                a_bce = -inputs["aln"] * jnp.log(aln + 1e-8)
                b_bce = -jax.nn.relu(1-inputs["aln"]) * jnp.log(jax.nn.relu(1-aln) + 1e-8)
                bce = (sm_mask * (a_bce+b_bce)).sum()
                loss += p["aln_lam"] * bce

            return x_ms_pred, loss

        if initialize_params: return _params, _model
        else: return _model
  #####################################################################################################
  #####################################################################################################

    def get_contacts(self, return_params=False):
        '''get contact map from W matrix'''
        def _apc(x):
            a1,a2 = x.sum(0,keepdims=True),x.sum(1,keepdims=True)
            y = x - (a1*a2)/x.sum()
            return y * (1-jnp.eye(x.shape[0]))

        # symmetrize and zero diag
        w = self.opt.get_params()["mrf"]["w"]
        #b = self.opt.get_params()["mrf"]["b"]
        w = (w + w.transpose([2,3,0,1]))/2
        w = w * (1-jnp.eye(self.p["x_ref_len"])[:,None,:,None])

        if return_params: return w #,b

        contacts = jnp.sqrt(jnp.square(w).sum((1,3)))
        return _apc(contacts)

    def get_auc(self):
        '''get contact accuracy'''
        contacts = np.array(self.get_contacts())
        #print(contacts)
        #print(self.nat_contacts)
        #print(self.nat_contacts_mask)
        return con_auc(pred=contacts,
                       true=self.nat_contacts,
                       mask=self.nat_contacts_mask)

    def get_aln(self, idx):
        '''get alignment'''
        inputs = {"x":self.X[idx], "lengths":self.lengths[idx],
                  "x_ref":self.X_ref, "key":[None]*3}
        if self.p["use_nat_aln"]: inputs["aln"] = self.nat_aln[idx]
        aln, sm = self.model_aln(self.opt.get_params(), inputs)
        return aln, sm
  
    def fit(self, steps=500, verbose=True):
        '''train model'''
        loss = 0
        for k in range(steps):
            idx = np.random.randint(0,self.X.shape[0],size=self.p["batch_size"])
            inputs = {"x":self.X[idx], "lengths":self.lengths[idx],
                    "x_ref":self.X_ref, "key":self.key.get(10)}
            if self.p["use_nat_aln"] or self.p["add_aln_loss"]: inputs["aln"] = self.nat_aln[idx]
            loss += self.opt.train_on_batch(inputs)
            if verbose and (k+1) % (steps//10) == 0:
                print_line,loss = [k+1, loss/(steps//10)],0
                if self.nat_contacts is not None: print_line.append(self.get_auc())
                if self.p["sw_learn_gap"]:
                    if self.p["sw_open"] is not None: print_line.append(self.opt.get_params()["open"])
                    if self.p["sw_gap"] is not None: print_line.append(self.opt.get_params()["gap"])
                if self.p["sw_learn_temp"]: print_line.append(self.opt.get_params()["temp"])
                print(*print_line)
    
    def reset_model_and_opt(self, new_p):
        for key in new_p:
            self.p[key]= new_p[key]
        model_params = self.opt.get_params()
        self.model = self._get_model(initialize_params=False)
        self.opt = laxy.OPT(self.model, model_params, lr=self.lr)
        
##############################################################
        
        
class BasicAlign:
    def __init__(self, X, lengths=None, batch_size=128, filters=512, win=18, 
               sw_unroll=4, sw_temp=1.0, sw_learn_temp=False,
               sw_open=None, sw_gap=None, sw_learn_gap=False,
               sw_restrict=False,
               seed=None, lr=0.1, norm_mode="fast", 
               w_scale=0.1, double_aln = False, double_frac = 0.0, pid_thresh = 1.0, supervise = False, nat_aln = None, pseudo= False):
    
        N,L,A = X.shape

        # inputs
        self.X = X
        self.lengths = X.sum([1,2]).astype(int) if lengths is None else lengths
        self.lr = lr*jnp.log(batch_size)/jnp.mean(self.lengths)
        self.nat_aln = nat_aln
        # seed for weight initialization and sub-sampling input
        self.key = laxy.KEY(seed)

        # params
        self.p = {"N":N, "L":L, "A":A, "batch_size":batch_size,
                  "sw_temp":sw_temp,"sw_learn_temp":sw_learn_temp,"sw_unroll":sw_unroll,
                  "sw_open":sw_open,"sw_gap":float(sw_gap),"sw_learn_gap":sw_learn_gap,
                   "filters":filters, "win":win,
                  "norm_mode":norm_mode,"w_scale":w_scale, "double_frac":double_frac, 
                  "double_aln":double_aln, "pid_thresh":pid_thresh, "supervise":supervise, "pseudo":pseudo}

        # initialize model
        self.init_params, self.model = self._get_model()

        self.model_aln = jax.jit(self._get_model(initialize_params=False, return_aln=True))
        self.opt = laxy.OPT(self.model, self.init_params, lr=self.lr)

  #####################################################################################################
    def _get_model(self, initialize_params=True, return_aln=False):
        p = self.p
        #######################
        # initialize params
        #######################
        if initialize_params:
            _params = {}
            _params["emb"] = Conv1D_custom()(p["A"],p["filters"],p["win"],key=self.key.get())
            _params["open"] = p["sw_open"]
            _params["gap"] = p["sw_gap"]
            _params["temp"] = p["sw_temp"]
            _params["msa"] = self.X[0,...]

        # get alignment
        def get_aln(z, lengths, gap=None, open=None, temp=1.0):
            # local-alignment (smith-waterman)
            if gap is None:
                aln_app = sw.sw_nogap(batch=True, unroll=p["sw_unroll"])
                aln = aln_app(z, lengths, temp)
            elif open is None:
                aln_app = sw.sw(batch=True, unroll=p["sw_unroll"])
                aln = aln_app(z, lengths, gap, temp)
            else:
                aln_app = sw.sw_affine(restrict_turns=True, batch=True, unroll=p["sw_unroll"])
                aln = aln_app(z, lengths, gap, open, temp)
            return aln

        #######################
        # setup the model
        #######################
        def _model(params, inputs):      
            
            #embed sequences and align to first seq/MSA embedding; this is sm_mtx is fed to SW
            emb = Conv1D_custom(params["emb"])(inputs["x"],key=inputs["key"],scale=p["w_scale"])
            sm_mtx = emb @ emb[0].T
            
            # mask
            lengths = jnp.stack([inputs["lengths"],
                               jnp.broadcast_to(inputs["lengths"][0],inputs["lengths"].shape)],-1)
            sm_mask_a = jnp.arange(p["L"]) < inputs["lengths"][:,None]
            sm_mask_b = jnp.arange(p["L"]) < inputs["lengths"][0]
            sm_mask = sm_mask_a[:,:,None] * sm_mask_b

            # normalize rows/cols (to remove edge effects due to 1D-convolution)
            sm_mtx = norm_row_col(sm_mtx, sm_mask, p["norm_mode"])
            
            if p["pseudo"]:
                aln = jnp.sqrt(jax.nn.softmax(sm_mtx, axis=-1) * jax.nn.softmax(sm_mtx, axis=-2))
            else:
                #settings for smith waterman
                sm_open = params["open"] if p["sw_learn_gap"] else laxy.freeze(params["open"])
                sm_gap = params["gap"] if p["sw_learn_gap"] else laxy.freeze(params["gap"])
                sm_temp = params["temp"] if p["sw_learn_temp"] else laxy.freeze(params["temp"])

                #align and compute MSA
                aln = get_aln(sm_mtx, lengths, gap=sm_gap, open=sm_open, temp=sm_temp)
            
            if return_aln:
                return aln, sm_mtx
            
            x_msa = jnp.einsum("nia,nij->nja", inputs["x"], aln)
            x_msa_bias = x_msa.mean(0)
      
            # compute MSA 
            if self.p["double_aln"] != False:
                if p["pid_thresh"]<=1.0:
                    pid  = jnp.einsum('nla,la->n', x_msa, x_msa[0,...])/ x_msa.shape[1]
                    x_msa_restricted = jnp.einsum('nia,n->nia',x_msa, (pid > p["pid_thresh"]))
                    num_surviving_seqs = (pid > p["pid_thresh"]).sum() +1
                    x_msa_bias_restricted = (inputs["x"][0,...]+x_msa_restricted.sum(axis = 0))/num_surviving_seqs
                else:
                    x_msa_bias_restricted = x_msa_bias
                ref = Conv1D_custom(params["emb"])(x_msa_bias_restricted[None,...],key=inputs["key"],scale=p["w_scale"])
                ref = ref[:self.lengths[0]][0,...]
                sm_mtx = emb @ ( p["double_frac"] * ref.T + (1-p["double_frac"])* emb[0].T )

                # normalize rows/cols (to remove edge effects due to 1D-convolution)
                sm_mtx = norm_row_col(sm_mtx, sm_mask, p["norm_mode"])

                aln = get_aln(sm_mtx, lengths, gap=sm_gap, open=sm_open, temp=sm_temp)
                x_msa = jnp.einsum("nia,nij->nja", inputs["x"], aln)
                x_msa_bias = x_msa.mean(0)
            
            #compute predictions based on composition of each column
            x_ms_pred = jnp.einsum("ja,nij->nia", x_msa_bias, aln)
            if p["supervise"]== True:
                true_msa = jnp.einsum("nia,nij->nja", inputs["x"], inputs["nat_aln"])
                doomed_cols = inputs["nat_aln"][0,...].sum(axis = 0)
                true_msa_r = (true_msa.transpose([0,2,1])*doomed_cols).transpose([0,2,1])
                
                x_msa_r = jnp.einsum("nla,lq->nqa", x_msa, inputs["nat_aln"][0,...])
                
                diff = -(true_msa_r * jnp.log(x_msa_r + 1e-8))
            else:
                diff = inputs["x"] * jnp.square(inputs["x"]-x_ms_pred)
            loss = diff.sum() 
                 
            return None, loss

        if initialize_params: return _params, _model
        else: return _model
    
  #####################################################################################################  
   
    def fit(self, steps=500, verbose=True):
        '''train model'''
        L=self.opt.fit(inputs={"x":self.X, "lengths":self.lengths, "nat_aln":self.nat_aln},
                    batch_size=self.p["batch_size"], steps=steps, return_losses=True)
        return L
    
  #####################################################################################################  
    
    def reset_model_and_opt(self, new_p):
        for key in new_p:
            self.p[key]= new_p[key]
        model_params = self.opt.get_params()
        self.model = self._get_model(initialize_params=False)
        self.opt = laxy.OPT(self.model, model_params, lr=self.lr)
