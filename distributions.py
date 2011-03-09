import numpy as np
from theano import tensor as T
from model import add_stochastic

def add_normal(model, name, m, v):
    new_var = model['stream'].normal(avg=m, std=T.sqrt(v))
    new_factors = [T.log(2*np.pi), 
                    -T.prod(T.shape(new_var))*T.log(v)/2,
                    -(new_var-m)**2/2/v]
    return add_stochastic(model, name, new_var, new_factors)