from __init__ import *
import time

m = empty_model()
# v, m1 = add_deterministic(m,'v',T.constant(3))
m1 = m
v=3
x, m2 = add_normal(m1,'x',0,v)
z, m3 = add_deterministic(m2,'z',x+3)
y, m4 = add_normal(m3,'y',z,100)

f = logp(m4)
f2 = logp(m4, [x])
f3 = simulate_prior(m4)
f4 = simulate_prior(m4, [x])
f5 = logp(m2, [x])

x2 = shallow_variable_copy(x,'x2')
f6 = logp_difference(m4, {x:x2}, compile=True)

n_iter = 2000

n_cycles_per_sweep = 10
n_sweeps = n_iter / n_cycles_per_sweep
out, t = mcmc(m4, {y:2}, n_sweeps, n_cycles_per_sweep)
out = full_trace(m4, out)

print 'Theano time: %f'%t

import pymc as pm
import time
v=3
x = pm.Normal('x',0,v)
z = x+3
y = pm.Normal('y',z,100,value=2,observed=True)
M = pm.MCMC([x,y,z])
t = time.time()
M.sample(n_iter)
t = time.time()-t
print 'PyMC time: %f'%t