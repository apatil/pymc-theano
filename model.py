import theano as th
import functools
from theano import tensor as T
import pysistence as ps
from pysistence import func

# A model is a collection of variables and factors.
# Variables 

def empty_model():
    return ps.make_dict(variables=ps.make_list(), factors=ps.make_list(T.constant(0)), stream=th.tensor.shared_randomstreams.RandomStreams())

def add_stochastic(model, name, new_variable, new_factors):
    new_variable.name = name
    return new_variable, model.using(variables = model['variables'].cons(new_variable),
                        factors = model['factors'].concat(ps.make_list(*new_factors)))

add_deterministic = functools.partial(add_stochastic, new_factors=[])
                        
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

def logp(model, arguments=[]):
    if any(map(isdeterministic, arguments)):
        raise TypeError, "The arguments of the logp function cannot include the deterministics %s."%filter(isdeterministic, arguments)
    return th.function(arguments, T.sum(list(model['factors'])), no_default_updates=True)

def to_namedict(values, variables):
    return ps.make_dict(**dict([(var.name, val) for var,val in zip(variables, values)]))
        
def simulate_prior(model, arguments=[], outputs = None):
    outputs = outputs or model['variables']
    if any(map(isdeterministic, arguments)):
        raise TypeError, "The arguments of the logp function cannot include the deterministics %s."%filter(isdeterministic, variables)
    return func.compose(th.function(arguments, list(outputs)), functools.partial(to_namedict, variables=outputs))
    
    