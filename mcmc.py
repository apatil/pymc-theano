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
    
    return info.using(acceptance=info.get('acceptance',0)+acc), 
            next_x, 
            state.replace(cur_x, next_x)

def compiled_mcmc_sweep(model, methods, info, n_cycles):
    "A declarative MCMC sweep. The entire sweep is compiled together, and returns to Python."
    state = model['variables']
    orig_variables = methods.keys()
    
    for i in xrange(n_cycles):
        new_variables = []
        for v,ov in zip(variables, orig_variables):
            info_, new_variable, state = methods[ov](model, state, ov, v, info[ov])
            info = info.using(ov=info_)
            new_variables.append(new_variable)
        variables = new_variables
    
    sweep_expression = list(state) + [info[ov] for ov in orig_variables]
    
    f = th.function(model['variables'], sweep_expression, no_default_updates=True)
    
    def sweep(init_state, f=f, orig_variables=orig_variables):
        raw_output = f(init_state)
        state_size =len(list(init_state))
        new_state = ps.make_list(*raw_output[:state_size])
        
        info = dict(zip(orig_variables, raw_output[state_size:]))
        
        return new_state, info
        
    return sweep
    
def tune(info, n_cycles):
    return info.using(acceptance=0)

# FIXME: Dataless submodels...    
def metropolis_mcmc(model, observations, n_sweeps, n_cycles_per_sweep, methods=empty_dict, info=empty_dict):
    "The full MCMC algorithm, which returns a trace."

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