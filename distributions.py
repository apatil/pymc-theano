import numpy as np
from theano import tensor as T
from model import add_stochastic, add_deterministic, neg_infinity

# TODO: Make 'exploding' logp and random functions that can stand in when the logp or random function
# is unknown.

def require(pred):
    return T.switch(pred, 0, neg_infinity)

def add_normal(model, name, m, v):
    new_var = model['stream'].normal(avg=m, std=T.sqrt(v))
    new_factors = [ require(T.gt(v,0)),
                    T.log(2*np.pi), 
                    -T.prod(T.shape(new_var))*T.log(v)/2,
                    -(new_var-m)**2/2/v]
    return add_stochastic(model, name, new_var, new_factors)