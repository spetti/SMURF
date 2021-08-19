import jax
import jax.numpy as jnp
import random
from jax.experimental.optimizers import adam

#################
# OPT function
#################

class OPT():
  def __init__(self, model, params, has_grad=False, lr=1e-3, optimizer=adam, seed=None):
    self._k = 0
    self.losses = []
    self._key = KEY(seed)
    self._opt_init, self._opt_update, self._opt_params = optimizer(step_size=lr)
    self._opt_state = self._opt_init(params)

    # split function into out, loss and grad
    self._fn_out = jax.jit(lambda p,i: model(p,i)[0])
    self._fn_loss = jax.jit(lambda p,i: model(p,i)[1])
    if has_grad: self._fn_grad = jax.jit(lambda p,i: model(p,i)[1:])
    else: self._fn_grad = jax.value_and_grad(lambda p,i: model(p,i)[1].sum())

    def update(k, state, inputs):      
      loss, grad = self._fn_grad(self._opt_params(state), inputs)
      state = self._opt_update(k, grad, state)
      return state, loss
    self._update = jax.jit(update)

  def train_on_batch(self, inputs):
    self._opt_state, loss = self._update(self._k, self._opt_state, self._add_key(inputs))
    self._k += 1
    return loss
    
  def set_params(self, params): self._opt_state = self._opt_init(params)
  def get_params(self): return self._opt_params(self._opt_state)
  def evaluate(self, inputs): return self._fn_loss(self.get_params(), self._add_key(inputs))
  def predict(self, inputs): return self._fn_out(self.get_params(), self._add_key(inputs))
  def grad(self, inputs): return self._fn_grad(self.get_params(), self._add_key(inputs))
  
  def fit(self, inputs, steps=100, batch_size=None, batch_inputs=None,
          verbose=True, return_losses=False, seed=None):
    if batch_size is not None:
      return self._fit_batch(inputs, steps, batch_size, batch_inputs,
                             verbose, return_losses, seed)
    for k in range(steps):
      loss = self.train_on_batch(inputs)
      if return_losses: self.losses.append(float(loss))
      if verbose and (k+1) % (steps//10) == 0: print(k+1, loss)
    if return_losses: return self.losses

  def _add_key(self, inputs):
    if isinstance(inputs, dict) and "key" not in inputs:
      return {**inputs,"key":self._key.get()}
    else: return inputs

  # TODO, clean up _fit_batch
  def _fit_batch(self, inputs, steps, batch_size, batch_inputs=None,
                 verbose=True, return_losses=False, seed=None):
    # spliting inputs into nonbatched and batched
    if batch_inputs is None: nonbatch_inputs, batch_inputs = type(inputs)(), inputs
    elif inputs is None: nonbatch_inputs = type(batch_inputs)()
    else: nonbatch_inputs = inputs
      
    N = len(jax.tree_util.tree_leaves(batch_inputs)[0])
    if N < batch_size:
      if verbose: print(f"WARNING: (N:{N} < batch_size:{batch_size})")
      return self.fit(inputs, steps, verbose=verbose, return_losses=return_losses)
    
    key, idx = KEY(seed=seed), jnp.arange(N)
    def subsample(inp, key):
      sub_idx = jax.random.choice(key, idx, shape=(batch_size,), replace=False)
      return jax.tree_util.tree_map(lambda x: x[sub_idx], inp)
    subsample = jax.jit(subsample)
    
    if verbose: loss_tot = 0
    for k in range(steps):
      inp = subsample(batch_inputs, key.get())
      if isinstance(inp, dict): inp.update(nonbatch_inputs)
      if isinstance(inp, list): inp += nonbatch_inputs
      if isinstance(inp, tuple): inp += nonbatch_inputs
      
      loss = self.train_on_batch(inp)
      if verbose: loss_tot += loss
      if return_losses: self.losses.append(float(loss))
      if verbose and (k+1) % (steps//10) == 0:
        print(k+1, loss_tot/(steps//10))
        loss_tot = 0
    if return_losses: return self.losses
      
#################
# UTILS
#################

def get_random_key(seed=None):
  '''get random key'''
  if seed is None: seed = random.randint(0,2147483647)
  return jax.random.PRNGKey(seed) 

def freeze(params):
  '''freeze provided parameters'''
  return jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(x), params)

def freeze_cond(cond, params):
  '''freeze provided parameters given [cond]ition'''
  return jax.lax.cond(cond, lambda _:freeze(params), lambda _:params, None)
  
class KEY():
  '''random key generator'''
  def __init__(self, seed=None):
    self.key = get_random_key(seed)    
  def get(self, num=1):
    if num > 1:
      self.key, *subkeys = jax.random.split(self.key, num=(num+1))
      return subkeys
    else:
      self.key, subkey = jax.random.split(self.key)
      return subkey

#################
# LAYERS
#################

def STAX(model, input_shape, key=None, seed=None):
  '''decompose stax model/layer(s) into params and model'''
  if key is None: key = get_random_key(seed)
  _init_params, _model = model
  _params = _init_params(key, input_shape)[1]
  return _params, _model

def MRF(params=None):
  '''
  markov random field layer
  ----------------------------------------------------
  params = MRF()(L=length, A=alphabet, use_bias=True)
  output = MRF(params)(input, return_w=False)
  '''
  def init_params(L, A, use_bias=True, key=None, seed=None):
    params = {"w":jnp.zeros((L,A,L,A))}
    if use_bias: params["b"] = jnp.zeros((L,A))
    return params
  
  def layer(x, return_w=False, rm_diag=True, symm=True, mask=None):
    w = params["w"]
    L,A = w.shape[:2]
    if rm_diag:
      # set diagonal to zero
      w = w * (1-jnp.eye(L)[:,None,:,None])
    if symm:
      # symmetrize
      w = 0.5 * (w + w.transpose([2,3,0,1]))
    if mask is not None:
      w = w * mask[:,None,:,None]      
      
    y = jnp.tensordot(x,w,2) # x (N,L,A), w (L,A,L,A)
    if "b" in params: y += params["b"]
      
    if return_w: return y,w
    else: return y

  if params is None: return init_params
  else: return layer

def Conv1D(params=None):
  '''
  1D convolutional layer
  ----------------------------------------------------
  params = Conv1D()(in_dims, out_dims, win, use_bias=True)
  output = Conv1D(params)(input, stride=1, padding="SAME")
  '''
  def init_params(in_dims, out_dims, win, use_bias=True, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    params = {"w":jax.nn.initializers.glorot_normal()(key,(out_dims,in_dims,win))}
    if use_bias: params["b"] = jnp.zeros(out_dims)
    return params
      
  def layer(x, stride=1, padding="SAME"):
    x = x.transpose([0,2,1])
    y = jax.lax.conv(x,params["w"],(stride,),padding=padding)
    y = y.transpose([0,2,1]) 
    if "b" in params: y += params["b"]
    return y
  
  if params is None: return init_params
  else: return layer

def Conv2D(params=None):
  '''
  2D convolutional layer
  ----------------------------------------------------
  params = Conv2D()(in_dims, out_dims, win, use_bias=True)
  output = Conv2D(params)(input, stride=1, padding="SAME")
  '''
  def init_params(in_dims, out_dims, win, use_bias=True, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    params = {"w":jax.nn.initializers.glorot_normal()(key,(out_dims,in_dims,win,win))}
    if use_bias: params["b"] = jnp.zeros(out_dims)
    return params
      
  def layer(x, stride=1, padding="SAME"):
    x = x.transpose([0,3,1,2]) # (batch, channels, row, col)
    y = jax.lax.conv(x,params["w"],(stride,stride),padding=padding) # (batch, filters, row, col)
    y = y.transpose([0,2,3,1]) # (batch, row, col, filters)
    if "b" in params: y += params["b"]
    return y
  
  if params is None: return init_params
  else: return layer

def Dense(params=None):
  '''
  dense or linear layer
  ----------------------------------------------------
  params = Dense()(in_dims, out_dims, use_bias=True)
  output = Dense(params)(input)
  '''
  def init_params(in_dims, out_dims, use_bias=True, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    params = {"w":jax.nn.initializers.glorot_normal()(key,(in_dims,out_dims))}
    if use_bias: params["b"] = jnp.zeros(out_dims)
    return params
  
  def layer(x):
    y = x @ params["w"]
    if "b" in params: y += params["b"]
    return y
  
  if params is None: return init_params
  else: return layer

def GRU(params=None):
  '''
  Gated recurrent unit (GRU) layer
  ----------------------------------------------------
  params = GRU()(in_dims, out_dims)
  output = GRU(params)(input)
  '''
  def init_params(in_dims, out_dims, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    k = jax.random.split(key, num=4)
    w_ini = jax.nn.initializers.glorot_normal()
    return {"zr":{"w":w_ini(k[0],(in_dims,out_dims,2)),
                  "u":w_ini(k[1],(out_dims,out_dims,2)),
                  "b":jnp.zeros((out_dims,2))},
            "h": {"w":w_ini(k[2],(in_dims,out_dims)),
                  "u":w_ini(k[3],(out_dims,out_dims)),
                  "b":jnp.zeros(out_dims)}}

  def layer(x):
    def gru_cell(h,x):
      p = params["zr"]
      tmp = jnp.tensordot(x,p["w"],[-1,0]) + jnp.tensordot(h,p["u"],[-1,0]) + p["b"]
      zt,rt = jax.nn.sigmoid(tmp).T
      ht = jnp.tanh(x@params["h"]["w"] + (h*rt)@params["h"]["u"] + params["h"]["b"])      
      h = (1-zt)*h + zt*ht
      return h,h

    out_dims = params["h"]["b"].shape[0]
    h0 = jnp.zeros(out_dims)
    h,seq = jax.lax.scan(gru_cell,h0,x)
    return seq

  if params is None: return init_params
  else: return jax.vmap(layer)

def LSTM(params=None):
  '''
  Long short-term memory (LSTM) layer
  ----------------------------------------------------
  params = LSTM()(in_dims, out_dims)
  output = LSTM(params)(input)
  '''
  def init_params(in_dims, out_dims, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    k = jax.random.split(key, num=2)
    w_ini = jax.nn.initializers.glorot_normal()
    return {"w":w_ini(k[0],(in_dims,out_dims,4)),
            "u":w_ini(k[1],(out_dims,out_dims,4)),
            "b":jnp.zeros((out_dims,4))}            
  
  def layer(x):
    def lstm_cell(hc,x):
      h,c = hc
      p = params
      tmp = jnp.tensordot(x,p["w"],[-1,0]) + jnp.tensordot(h,p["u"],[-1,0]) + p["b"]
      ft, it, ot, gt = tmp.T
      ct = jax.nn.sigmoid(ft + 1) * c + jax.nn.sigmoid(it) * jnp.tanh(gt)
      ht = jax.nn.sigmoid(ot) * jnp.tanh(ct)
      return (ht,ct),ct

    out_dims = params["b"].shape[0]
    h0 = jnp.zeros(out_dims)
    h,seq = jax.lax.scan(lstm_cell,(h0,h0),x)
    return seq

  if params is None: return init_params
  else: return jax.vmap(layer)
