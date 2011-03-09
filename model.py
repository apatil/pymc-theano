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
                        
def add_normal(model, name, m, v):
    new_var = model['stream'].normal(avg=m, std=T.sqrt(v))
    new_factors = [T.log(2*np.pi), 
                    -T.prod(T.shape(new_var))*T.log(v)/2,
                    -(new_var-m)**2/2/v]
    return add_stochastic(model, name, new_var, new_factors)

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
    
if __name__ == '__main__':
    m = empty_model()
    x, m2 = add_normal(m,'x',0,1)
    z, m3 = add_deterministic(m2,'z',x+3)
    y, m4 = add_normal(m3,'y',z,100)
    f = logp(m4)
    f2 = logp(m4, [x])
    seed(stochastics(m4), 3)
    print f()
    