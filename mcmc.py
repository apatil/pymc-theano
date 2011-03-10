import theano as th
from theano import tensor as T
import pysistence as ps
import numpy as np
import model

def metropolis(model, state, orig_x, cur_x, sigma):
    "A declarative Metropolis step replacing x in the state vector with its new value."
    x_p = model['stream'].normal(avg=cur_x, std=sigma)
    
    lpd = logp_difference(model, {orig_x:x_p}, compile=False)(state)
    acc = T.log(model['stream'].uniform())<lpd
    x_next = T.switch(acc, x_p, cur_x)
    
    return acc, x_next, state.replace(cur_x, x_next)

def metropolis_sweep(model, variables, sigmas, info, n_cycles):
    "A declarative Metropolis sweep. The entire sweep is compiled together, and returns to Python."
    state = model['variables']
    orig_variables = variables
    
    for i in xrange(n_cycles):
        new_variables = []
        for v,ov in zip(variables, orig_variables):
            acc, new_variable, state = metropolis(model, state, ov, v, sigmas[ov])
            new_variables.append(new_variable)
            info[ov] = info[ov]+acc
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
    
def tune(sigma, acceptance):
    return sigma    

# FIXME: Dataless submodels...    
def metropolis_mcmc(model, observations, sigmas, n_sweeps, n_cycles_per_sweep):
    "The full MCMC algorithm, which returns a trace."

    fixed_variables = []
    for v in model['variables']:
        if not observations.has_key(v) and isstochastic(v):
            variables_to_update.append(v)

    for v in variables_to_update:
        sigmas[v] = sigmas.get(v,1)
    
    # Make an initial state vector, in order.    
    state = simulate_prior(model, observations.keys())(*[observations[k] for k in observations.keys()])
    state = ps.make_list(*[state[v] for v in model['variables']])

    info = dict([(v,0) for v in variables_to_update])

    sweep_fn = metropolis_sweep(model, variables_to_update, sigmas, info, n_cycles_per_sweep)
    
    trace = [state]
    for i in xrange(n_sweeps):
        state, info = sweep_fn(trace[-1])
        trace.append(state)
        for v in variables_to_update:
            sigmas[v] = tune(sigmas[v], info[v])
            
    return trace