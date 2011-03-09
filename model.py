import theano as th
import functools
from theano import tensor as T
import pysistence as ps
from pysistence import func

# A model is a collection of variables and factors.
# Variables 

def empty_model():
    "Initializes a new empty model."
    return ps.make_dict(variables=ps.make_list(), factors=ps.make_list(T.constant(0)), stream=th.tensor.shared_randomstreams.RandomStreams())

def add_stochastic(model, name, new_variable, new_factors):
    "Returns a stochastic variable, and a version of model that incorporates that variable."
    new_variable.name = name
    return new_variable, model.using(variables = model['variables'].cons(new_variable),
                        factors = model['factors'].concat(ps.make_list(*new_factors)))

def add_deterministic(model, name, new_variable):
    "Returns a deterministic variable, and a version of the model that incorporates that variable."
    return add_stochastic(model, name, new_variable, new_factors)
                        
def isstochastic(variable):
    "Tests whether the variable is stochastic conditional on its parents. WARNING: this implementation may be too brittle."
    return hasattr(variable,'rng')
    
def isdeterministic(variable):
    "Tests whether the variable is deterministic conditional on its parents."
    return not isstochastic(variable)

def seed(variables, seed):
    "Seeds all the rngs used by the variables."
    [v.rng.value.seed(seed) for v in filter(isstochastic,variables)]

def stochastics(model):
    "Returns all the stochastic variables in the model."
    return filter(isstochastic, model['variables'])
    
def deterministics(model):
    "Returns all the deterministic variables in the model."
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
    
    