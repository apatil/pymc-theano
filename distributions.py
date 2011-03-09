def add_normal(model, name, m, v):
    new_var = model['stream'].normal(avg=m, std=T.sqrt(v))
    new_factors = [T.log(2*np.pi), 
                    -T.prod(T.shape(new_var))*T.log(v)/2,
                    -(new_var-m)**2/2/v]
    return add_stochastic(model, name, new_var, new_factors)

if __name__ == '__main__':
    m = empty_model()
    x, m2 = add_normal(m,'x',0,1)
    z, m3 = add_deterministic(m2,'z',x+3)
    y, m4 = add_normal(m3,'y',z,100)
    f = logp(m4)
    f2 = logp(m4, [x])
    seed(stochastics(m4), 3)
    print f()
