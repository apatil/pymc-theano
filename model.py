import theano as th
import functools
from theano import tensor as T
import pysistence as ps
from pysistence import func
import numpy as np

neg_infinity = T.constant(-np.inf)
# TODO: How to find dataless submodels? Some intermediate deterministics may not be added to the model.
# TODO: Can more graphical structure be shared with theano itself?

def empty_model():
    "Initializes a new empty model."
    return ps.make_dict(variables=ps.make_list(), factors=ps.make_list(T.constant(0)), stream=th.tensor.shared_randomstreams.RandomStreams())

def check_namedup(model, name):
    if name in [v.name for v in model['variables']]:
        raise ValueError, 'The model already contains a variable named %s.'%name

def add_stochastic(model, name, new_variable, new_factors):
    "Returns a stochastic variable, and a version of model that incorporates that variable."
    new_variable.name = name
    check_namedup(model, name)
    return new_variable, model.using(variables = model['variables'].cons(new_variable),
                        factors = model['factors'].concat(ps.make_list(*new_factors)))

def add_deterministic(model, name, new_variable):
    "Returns a deterministic variable, and a version of the model that incorporates that variable."
    new_variable.name = name
    check_namedup(model, name)
    return new_variable, model.using(variables = model['variables'].cons(new_variable))
            
def isstochastic(variable):
    "Tests whether the variable is stochastic conditional on its parents. WARNING: this implementation may be too brittle."
    return hasattr(variable,'rng')

def isconstant(variable):
    "Tests whether the variable is a Theano constant."
    return isinstance(variable, th.tensor.basic.TensorConstant)
    
def isdeterministic(variable):
    "Tests whether the variable is deterministic conditional on its parents."
    return not isstochastic(variable) and not isconstant(variable)

def seed(model, seed):
    "Seeds all the rngs used by the variables."
    model['stream'].seed(seed)

def stochastics(model):
    "Returns all the stochastic variables in the model."
    return filter(isstochastic, model['variables'])
    
def deterministics(model):
    "Returns all the deterministic variables in the model."
    return filter(isdeterministic, model['variables'])

def check_no_deterministics(arguments, fname):
    "Raises an error if any of arguments are deterministics"
    if any(map(isdeterministic, arguments)):
        raise TypeError, "The arguments of the function %s cannot include the deterministics %s."%(fname, filter(isdeterministic, arguments))

def logp_or_neginf(logps):
    """
    If any of the logps is -infinity, returns -infinity regardless of whether
    any others are infinity, nan or whatever. Otherwise, returns the sum of 
    the logps.
    """
    logps = list(logps)
    return T.switch(T.sum(T.eq(logps, neg_infinity)), neg_infinity, T.sum(logps))

def logp(model, arguments=None):
    """
    Returns a function that takes values for some stochastic variables in the model,
    and returns the total log-probability of the model given all the other variables'
    current values.
    """
    arguments = arguments or stochastics(model)
    check_no_deterministics(arguments, 'logp')
    return th.function(arguments, logp_or_neginf(model['factors']), no_default_updates=True)
    
def logp_gradient(model, wrt, arguments=None):
    """
    Returns a function that takes values for some stochastic variables in the model,
    and returns the gradient of the total log-probability with respect to the variables
    in 'wrt'.
    """
    arguments = arguments or stochastics(model)
    check_no_deterministics(arguments, 'logp_gradient')
    return th.function(arguments, T.grad(logp_or_neginf(model['factors']), wrt), no_default_updates=True)

def get_argument_names(f):
    "Returns the names of the arguments of the Theano function."
    return [s.name for s in f.input_storage]

def logp_difference(model, wrt, arguments=None):
    """
    Returns a function that takes current values for some stochastic variables in the model,
    and new values for some stochastic variables in the model, and returns the new logp minus
    the current logp.
    """
    raise NotImplementedError, "this should work yet, but it doesn't."
    all_stochastics = stochastics(model)
    arguments = arguments or all_stochastics
    check_no_deterministics(arguments, 'logp_difference')
    check_no_deterministics(wrt, 'logp_difference')
    
    replacements = {}
    for w in wrt:
        replacements[w] = shallow_variable_copy(w, w.name+'_new')
    differences = []
    for f in model['factors']:
        f_new = th.clone(f, replace=replacements)
        differences.append(f_new - f)

    return th.function(arguments + [replacements[w] for w in wrt], logp_or_neginf(differences), no_default_updates=True)

def to_namedict(variables, values):
    "Makes a persistent dict mapping variable name to value."
    return ps.make_dict(**dict([(var.name, val) for var,val in zip(variables, values)]))
        
def simulate_prior(model, arguments=[], outputs = None):
    """
    Returns a function that takes values for some stochastic variables in the model,
    and simulates the rest of the model from its conditional prior. By default, all
    the model's variables' simulated values are returned in a dict. Optionally, only
    some variables' values can be computed and returned.
    """
    outputs = outputs or model['variables']
    if any(map(isdeterministic, arguments)):
        raise TypeError, "The arguments of the logp function cannot include the deterministics %s."%filter(isdeterministic, variables)
    f__ = th.function(arguments, list(outputs))
    def f_(*values):
        return to_namedict(outputs, f__(*values))
    return f_
    
    