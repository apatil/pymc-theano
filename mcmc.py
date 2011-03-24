import theano as th
from theano import tensor as T
import pysistence as ps
import numpy as np
from model import *
import time

empty_dict = ps.make_dict()

def zipmap(seq, f):
    return ps.make_dict(**dict([(s,f(s)) for s in seq]))

def index_plist(pl, i):
    return list(pl)[i]
    
def remember(model):
    return simulate_prior(model, arguments=stochastics(model))
    
def full_trace(model, trace):
    r = remember(model)
    return ps.make_list(*[r(state) for state in trace])

class metropolis(object):
    def __init__(self, model, orig_x):
        self.x_p = orig_x + 1
        self.lpd_ = logp_difference(model, {orig_x:self.x_p}, compile=False)
    
    def lpd(self, state, x_p):
        return self.lpd_(state.concat(ps.make_list(x_p)))
    
    @classmethod
    def tune(cls, info_state):
        raise NotImplementedError
    
    @classmethod
    def init_tuning_info(cls, x):
        raise NotImplementedError

    def step(self, model, state, cur_x, orig_x):
        "A declarative Metropolis step replacing x in the state vector with its new value."

        x_p = model['stream'].normal(avg=cur_x, std=1)
        lpd = self.lpd(state, x_p)
        acc = T.log(model['stream'].uniform())<lpd

        next_x = T.switch(acc, x_p, cur_x)
        next_x.name = cur_x.name
        
        out= state.replace(cur_x, next_x)
        return out

def compiled_mcmc_sweep(model, methods, n_cycles):
    """
    A declarative MCMC algorithm, running for a set number of cycles.
    Only the terminal value is returned to Python.
    """
    
    variables_to_update = methods.keys()
    state = ps.make_list(*stochastics(model))
    orig_state = list(state)
    
    state_index = [orig_state.index(v) for v in variables_to_update]
    
    for i in xrange(n_cycles):
        for v, index in zip(variables_to_update, state_index):
            state = methods[v].step(model, state, index_plist(state, index), v)
    
    f = th.function(orig_state, list(state), no_default_updates=True, mode='FAST_RUN')
    # th.printing.pydotprint(f,'f.pdf')
    
    def sweep(state_value_dict, orig_state=orig_state, f=f):
        "Takes a state, represented as a dict, applies n_cycles MCMC steps to it, and returns a new state represented as a dict."
        new_state_value_list = f(**state_value_dict)
        return to_namedict(orig_state, new_state_value_list)
    
    return sweep
    
# FIXME: Dataless submodels...    
def mcmc(model, observations, n_sweeps, n_cycles_per_sweep, methods=empty_dict, seed=None):
    "The full MCMC algorithm, which returns a trace."

    if seed:
        seed_model(model, seed)
        
    s = stochastics(model)

    # Make an initial state vector, in order.
    observation_value_dict = dict([(ov.name, observations[ov]) for ov in observations.keys()])
    state_value = simulate_prior(model, outputs=s, arguments=observations.keys())(observation_value_dict)

    variables_to_update = []
    for v in s:
        if not observations.has_key(v):
            variables_to_update.append(v)

    methods = zipmap(variables_to_update, lambda v: methods.get(v, metropolis(model, v)))

    sweep_fn = compiled_mcmc_sweep(model, methods, n_cycles_per_sweep)
    
    trace = ps.make_list(state_value)
    
    t = time.time()
    for i in xrange(n_sweeps):
        state_value = sweep_fn(state_value)
        trace = trace.cons(state_value)
    t = time.time()-t
                    
    return trace, t