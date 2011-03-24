import theano as th
from theano import tensor as T
import pysistence as ps
import numpy as np
from model import *

empty_dict = ps.make_dict()

def zipmap(seq, f):
    return ps.make_dict(**dict([(s,f(s)) for s in seq]))

class metropolis(object):
    
    @classmethod
    def tune(cls, info_state):
        return info_state
    
    @classmethod
    def init_info(cls, x):
        acc = T.zeros(T.shape(x))
        acc.name = 'acceptance'
        sig = T.ones(T.shape(x))
        sig.name = 'sigma'
        return ps.make_dict(acceptance=acc, sigma=sig)

    @classmethod
    def step(cls, model, state, orig_x, cur_x, info):
        "A declarative Metropolis step replacing x in the state vector with its new value."
        sigma = info.get('sigma',1)
        x_p = model['stream'].normal(avg=cur_x, std=sigma)
    
        lpd = logp_difference(model, {orig_x:x_p}, compile=False)
        acc = T.log(model['stream'].uniform())<lpd
        next_x = T.switch(acc, x_p, cur_x)
        next_x.name = cur_x.name
        next_acc = info.get('acceptance',0)+acc
        next_acc.name = 'acceptance'
    
        return info.using(acceptance=next_acc), \
                next_x, \
                state.replace(cur_x,next_x)

def get_info_state(info):
    infostate = {}
    for k,v in info.iteritems():
        infostate_ = {}
        for k_, v_ in v.iteritems():
            infostate_[k_] = v_.get_constant_value()
        infostate[k] = infostate_
    return ps.make_dict(**infostate)
    
def flatten_info(state, info, variables):
    flat_input = list(state)
    for v in variables:
        flat_input += info[v].values()
    return flat_input
    
def unflatten_info(flat_output, old_state, old_info, variables):
    len_oldstate = len(list(old_state))
    info = {}
    counter = len_oldstate
    for v in variables:
        keys = old_info[v].keys()
        info[v] = zipmap(keys, lambda k: flat_output[counter:len(keys)])
        # ps.make_dict(**dict(zip(keys, flat_output[counter:len(keys)])))
        counter += len(keys)
    return ps.make_list(*flat_output[:len_oldstate]), ps.make_dict(**info)

def compiled_mcmc_sweep(model, variables_to_update, methods, info, n_cycles):
    """
    A declarative MCMC algorithm, running for a set number of cycles.
    Only the terminal value is returned to Python.
    """
    orig_info = info
    orig_state = model['variables']
    state = orig_state
    info = dict(info)

    variables = dict([(v,v) for v in variables_to_update])
    for i in xrange(n_cycles):
        for ov,v in variables.items():
            info[ov], variables[ov], state = methods[ov].step(model, state, ov, v, info[ov])
    new_variables_to_update = [variables[ov] for ov in variables_to_update]
    
    import pdb
    pdb.set_trace()
    flat_input = flatten_info(orig_state, orig_info, variables_to_update)
    flat_output = flatten_info(state, info, variables_to_update)
    f = th.function(flat_input, flat_output, no_default_updates=True)
    
    th.printing.pydotprint(f,'mcmc.png')
    
    def sweep(state, info, f=f, orig_variables=variables_to_update, variables = new_variables_to_update):
        "Maps (state, info) to (state, info) after %i MCMC cycles."%n_cycles
        flat_input = flatten_info(state, info, orig_variables)
        flat_output = f(flat_input)        
        return unflatten_info(flat_output, state, info, variables)
        
    return sweep
    
def tune(info, n_cycles):
    return info.using(acceptance=0)

def init_info(variables, methods, info=empty_dict):
    # return ps.make_dict(**dict([(v, info.get(v, methods[v].init_info(v))) for v in variables]))
    return zipmap(variables, lambda v: info.get(v,methods[v].init_info(v)))


# FIXME: Dataless submodels...    
def mcmc(model, observations, n_sweeps, n_cycles_per_sweep, methods=empty_dict, info=empty_dict, seed=None):
    "The full MCMC algorithm, which returns a trace."

    if seed:
        seed_model(model, seed)

    variables_to_update = []
    for v in model['variables']:
        if not observations.has_key(v) and isstochastic(v):
            variables_to_update.append(v)

    methods = zipmap(variables_to_update, lambda v: methods.get(v, metropolis))
    # ps.make_dict(**dict([(v, methods.get(v, metropolis)) for v in variables_to_update]))
    info = init_info(variables_to_update, methods, info)
    sweep_fn = compiled_mcmc_sweep(model, variables_to_update, methods, info, n_cycles_per_sweep)
    
    # Make an initial state vector, in order.
    observation_value_dict = dict([(ov.name, observations[ov]) for ov in observations.keys()])
    s = simulate_prior(model, outputs=model['variables'], arguments=observations.keys())(observation_value_dict)
    state = [s[v.name] for v in model['variables']]
    info_state = get_info_state(info)
    
    trace = ps.make_list(state)
    tuning_trace = ps.make_list(info_state)
    for i in xrange(n_sweeps):
        state, info_state = sweep_fn(state, info_state)
        trace.cons(state)
        tuning_trace.cons(info_state)
        for v in variables_to_update:
            info_state[v] = methods[v].tune(info_state[v])
            
    return trace, tuning_trace