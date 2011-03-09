import numpy as np
import theano as th
from theano import tensor as T
import pysistence as ps

# A model is a collection of variables and factors.
# Variables 

def empty_model():
    return ps.make_dict(variables=ps.make_list(), factors=ps.make_list(T.constant(0)), stream=th.tensor.shared_randomstreams.RandomStreams())

def add_stochastic(model, name, new_variable, new_factors):
    new_variable.name = name
    return new_variable, model.using(variables = model['variables'].cons(new_variable),
                        factors = model['factors'].concat(ps.make_list(*new_factors)))

def add_deterministic(model, name, new_variable):
    return add_stochastic(model, name, new_variable, [])
                        
def isstochastic(variable):
    return hasattr(variable,'rng')
    
def isdeterministic(variable):
    return not isstochastic(variable)

def seed(variables, seed):
    [v.rng.value.seed(seed) for v in variables]

def stochastics(model):
    return filter(isstochastic, model['variables'])
    
def deterministics(model):
    return filter(isdeterministic, model['variables'])

def logp(model, variables=None):
    if variables is None:
        return th.function([], T.sum(list(model['factors'])), no_default_updates=True)
    else:
        if any(map(isdeterministic, variables)):
            raise TypeError, "The arguments of the logp function cannot include the deterministics %s."%filter(isdeterministic, variables)
        return th.function(variables, T.sum(list(model['factors'])), no_default_updates=True)
    