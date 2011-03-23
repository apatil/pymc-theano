import theano as th
from theano import tensor as T
import pysistence as ps
import numpy as np
import model

empty_dict = ps.make_dict()

def metropolis(model, state, orig_x, cur_x, info):
    "A declarative Metropolis step replacing x in the state vector with its new value."
    sigma = info.get('sigma',1)
    x_p = model['stream'].normal(avg=cur_x, std=sigma)
    
    lpd = logp_difference(model, {orig_x:x_p}, compile=False)(state)
    acc = T.log(model['stream'].uniform())<lpd
    next_x = T.switch(acc, x_p, cur_x)
    
    return info.using(acceptance=info.get('acceptance',0)+acc), \
            next_x, \
            state.replace(cur_x, next_x)

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
        info[v] = ps.make_dict(**dict(zip(keys, flat_output[counter:len(keys)])))
        counter += len(keys)
    return ps.make_list(*flat_output[:len_oldstate]), ps.make_dict(**info)
    

def compiled_mcmc_sweep(model, methods, info, n_cycles):
    """
    A declarative MCMC algorithm, running for a set number of cycles.
    Only the terminal value is returned to Python.
    """
    state = model['variables']
    orig_variables = methods.keys()
    variables = orig_variables
    orig_info = info
    
    for i in xrange(n_cycles):
        new_variables = []
        for v,ov in zip(variables, orig_variables):
            info_, new_variable, state = methods[ov](model, state, ov, v, info[ov])
            info = info.using(ov=info_)
            new_variables.append(new_variable)
        variables = new_variables
    
    flat_input = flatten_info(model['variables'], orig_info, orig_variables)
    flat_output = flatten_info(state, info, variables)
    f = th.function(flat_input, flat_output, no_default_updates=True)
    
    def sweep(state, info, f=f, orig_variables=orig_variables):
        "Maps (state, info) to (state, info) after %i MCMC cycles."%n_cycles
        flat_input = flatten_info(state, info, orig_variables)
        flat_output = f(state)        
        return unflatten_info(flat_output, state, info, orig_variables)
        
    return sweep
    
def tune(info, n_cycles):
    return info.using(acceptance=0)

# FIXME: Dataless submodels...    
def mcmc(model, observations, n_sweeps, n_cycles_per_sweep, methods=empty_dict, info=empty_dict, seed=None):
    "The full MCMC algorithm, which returns a trace."

    if seed:
        seed_model(model, seed)

    fixed_variables = []
    for v in model['variables']:
        if not observations.has_key(v) and isstochastic(v):
            variables_to_update.append(v)

    methods = ps.make_dict(**dict([(v: methods.get(v, metropolis)) for v in variables_to_update]))
    info = ps.make_dict(**dict([(v: info.get(v, empty_dict)) for v in variables_to_update]))

    # Make an initial state vector, in order.    
    state = simulate_prior(model, observations.keys())(*[observations[k] for k in observations.keys()])
    state = ps.make_list(*[state[v] for v in model['variables']])

    info = dict([(v,0) for v in variables_to_update])

    sweep_fn = compiled_mcmc_sweep(model, methods, info, n_cycles_per_sweep)
    
    trace = ps.make_list(state)
    tuning_trace = ps.make_list(info)
    for i in xrange(n_sweeps):
        state, info = sweep_fn(trace.first)
        trace.cons(state)
        tuning_trace.cons(info)
        for v in variables_to_update:
            sigmas[v] = tune(sigmas[v], info[v])
            
    return trace, tuning_trace