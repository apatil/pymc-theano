import theano as th
from theano import tensor as T
import pysistence as ps
import numpy as np
import model

def state(model):
    v = th.function([], list(model['variables']))
    f = th.function([], list(model['factors']))
    