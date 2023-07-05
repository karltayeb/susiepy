#%%
from ibss2m import add_re, ibss2m_tilted
from ser import fit_tilted_ser, fit_tilted_ser2, initialize_re, initialize_ser_params, initialize_univariate_params
import numpy as np
from tilted import dict2param, param2dict, xi_fixed_point
from tilted import tilted_obj, tilted_univariate_lr, tilted_elbo
from utils import plot_contour, plot_1d

def simulate_data(n, p):
    X = np.random.normal(size=n*p).reshape(n, -1)
    y = np.random.binomial(1, 1/(1 + np.exp(1 - 2* X[:, :1].sum(1))), n)
    return dict(X=X, y=y)

def get_data_x(data, idx):
    return dict(X = data['X'][:, idx], y=data['y'])

# simulate data
n = 1000
p = 500
re = initialize_re(n, mu2=1)
params = initialize_ser_params(n, p, 1.0)
data = simulate_data(n, p)

#%%
# fit single
tau0 = 1.
mu2 = 0.

from jax import jit, vmap
from ser import initialize_univariate_params
from tilted import tilted_univariate_lr
params = initialize_univariate_params(n, tau0)
re = initialize_re(n, mu2=mu2)
data = simulate_data(n, p)
data = get_data_x(data, 0)
res = tilted_univariate_lr(data, re, params)
res['params']['mu']
#%% 
# plot mu
from tilted import tilted_bound

mu_range = (1, 5, 50)
var_range = (0.0001, 0.1, 50)
delta_range = (-2, 2, 50)

# plot tilted bound as a function of mu
def mu_bound(mu):
    params = dict(mu=mu, var=0.1, delta=-1)
    params2 = dict(tau0=1, xi = np.ones(n) * 0.5)
    re = initialize_re(n, mu2=5)
    params2['xi'] = xi_fixed_point(data, re, params, params2)
    return tilted_elbo(data, re, params, params2)
ff = vmap(mu_bound, 0, 0)
plot_1d(ff, (0, 20, 50))

#%%
# plot var
from utils import normal_kl
def var_bound(var):
    params = dict(mu=2, var=var, delta=-1)
    params2 = dict(tau0=10, xi = np.ones(n) * 0.5)
    params2['xi'] = xi_fixed_point(data, re, params, params2)
    #return normal_kl(params['mu'], params['var'], 0, 1/params2['tau0'])
    return tilted_elbo(data, re, params, params2)
ff = vmap(var_bound, 0, 0)
plot_1d(ff, (0.001, 0.1, 50))

#%%
# contour plot beta, sigma
# objective function with mu, var
@jit
def mu_var_contour(mu, var):
    params = dict(mu=mu, var=var, delta=1)
    params2 = dict(tau0=1, xi = np.ones(n) * 0.5)
    # make bound tight for this setting of parameters
    params2['xi'] = xi_fixed_point(data, re, params, params2)
    return tilted_elbo(data, re, params, params2)
ff = vmap(vmap(mu_var_contour, (0, 0), 0), (1, 1), 1)
plot = plot_contour(ff, mu_range, var_range)

#%%
# objective function with mu, delta
@jit
def mu_delta_contour(mu, delta):
    params = dict(mu=mu, var=0.1, delta=delta)
    params2 = dict(tau0=1, xi = np.ones(n) * 0.5)
    # make bound tight for this setting of parameters
    params2['xi'] = xi_fixed_point(data, re, params, params2)
    return tilted_elbo(data, re, params, params2)
ff = vmap(vmap(mu_delta_contour, (0, 0), 0), (1, 1), 1)
plot = plot_contour(ff, mu_range, delta_range)

#%%
# plot xi
# upper bound of <log(1 + exp(x)> is minimized at a unique point on [0, 1]
# i.e. lower bound to <-log(1 + exp(x))> is maximized

def plot_xi_1d(mu, var):
    from utils import normal_kl
    def bound(mu, var, xi):
        tmp = mu + 0.5 * (1 - 2 * xi) * var
        return 0.5 * xi**2 * var + jnp.log(1 + jnp.exp(tmp))

    ff = lambda xi: bound(mu, var, xi)
    plot_1d(ff, (0.0, 1.0, 50))

plot_xi_1d(1, 1)
plot_xi_1d(1, 10)
plot_xi_1d(1, 100)
plot_xi_1d(-3, 0.1)

#%% fixed point plot
def plot_xi_fixed(mu, var, alpha=0):
    f = lambda xi: sigmoid(mu + 0.5 * (1 - 2 * xi) * var)
    f = dampen(f, alpha)
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, color='red', linestyle='dotted')
    plt.plot(x, f(x), color='k')

plot_xi_fixed(0, 1, 0)
plot_xi_fixed(0, 1, 0.9)
plt.show()
plt.close()

plot_xi_fixed(0, 100, 0)
plot_xi_fixed(0, 1000, 0.9)
plot_xi_fixed(0, 1000, 0.99)

plt.show()
plt.close()


#%%


#%%
# Contour plot x=mu y=var z= fixed point of xi

from utils import sigmoid
from jax.lax import fori_loop
import jax.numpy as jnp
from jax import jit, vmap

def check_diff(val):
    # not_converged = jnp.logical_and(
    #     val['diff'] > val['tol'],
    #     val['it'] <= val['maxit']
    # )
    not_converged = val['diff'] > val['tol']
    return not_converged

def while_loop2(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val

def fori_loop2(lower, upper, body_fun, init_val):
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val

def while_not_converged(f, init, tol=1e-3, maxit=100):
    control = dict(tol=tol, maxit=maxit)
    def body_fun(val):
        y = f(val['y'])
        val = dict(
            y=y,
            diff = jnp.abs(y - val['y']),
            it = 1 + val['it'])
        val.update(control)
        return val
    val_init = dict(y=init, diff=tol+1, it=0)
    val_init.update(control)
    res = while_loop2(check_diff, body_fun, val_init)
    return res['y']

def fixed_point_iter(mu, var, xi):
    # update xi
    tmp = mu + 0.5 * (1 - 2 * xi) * var
    xi = sigmoid(tmp)
    return xi

def fixed_point_path(mu, var):
    xi = [0.1]
    for _ in range(100):
        xi.append(fixed_point_iter(mu, var, xi[-1]))
        print(f'bound: {bound(mu, var, xi[-1])}, xi = {xi[-1]}')

# def fixed_point_xi(mu, var):
#     f = lambda xi: fixed_point_iter(mu, var, xi)
#     return while_not_converged(f, 0.5)

def fixed_point_xi(mu, var):
    f = lambda i, xi: fixed_point_iter(mu, var, xi)
    return fori_loop(0, 100, f, 0.5)

print(fixed_point_xi(1, 0.1))

fixed_point_xi_vec = vmap(vmap(fixed_point_xi, (0, 0), 0), (1, 1), 1)
plot_contour(fixed_point_xi_vec, (-5, 5, 50), (0, 10, 50))

#%%
# Impliment Steffenson algorithm to stabilize fixed point iteration
def g(f, x, fx):
    return lambda x: f(x + fx)/fx - 1

# Yield iterator for Steffenson alg
def steff(f, x):
    while True:
        fx = f(x)
        gx = g(f, x, fx)(x)
        if gx == 0:
            break
        else:
            x = x - fx/gx
            yield x

mu = 1
var = 50
F = lambda xi: fixed_point_iter(mu, var, xi) - xi

steffF = steff(F, 0.01)
for _ in range(1000):
    next(steffF)
for _ in range(10):
    print(next(steffF))


#%%
def running_average(g):
  sum = 0
  count = 0
  while True:
    sum += next(g)
    count += 1
    yield sum/count

mu = 1
var = 200
steffF = steff(F, 0.5)
averageF = running_average(steffF)
for _ in range(10000):
    next(averageF)

for _ in range(10):
    print(next(averageF))


#%%

# Damped fixed point iteration to deal with cycling?
# x = f(x) <-> x = af(x) + (1 - a)x
# linear term does not impact convexity (f'' < 0)
def dampen(f, alpha=0.5):
    return lambda x: (1-alpha)*f(x) + alpha*x

mu = 1
var = 100

f = lambda xi: fixed_point_iter(mu, var, xi)
f_damp = dampen(f)

reporter = lambda xi: print(f'bound = {tilted_bound(mu, var, xi)}, xi = {xi}')

xi = 0.5
reporter(xi)
for _ in range(10):
    xi = f_damp(xi)
    reporter(xi)


#%%
# 1d newton optimization
from jax import grad
from utils import sigmoid
from jax.lax import fori_loop
import jax.numpy as jnp
from jax import jit, vmap

def check_diff(val):
    not_converged = val['diff'] > val['tol']
    return not_converged

def newton_step(x, H, g, t):
    return x - t * H(x) * g(x)

def newton_1d(f):
    g = grad(f)
    H = grad(g)
    step = lambda x: x - H(x)*g(x)
    optimizer = lambda x: while_not_converged(step, x, tol=1e-5, maxit=100)
    return optimizer

def tilted_bound(mu, var, xi):
    tmp = mu + 0.5 * (1 - 2 * xi) * var
    return 0.5 * xi**2 * var + jnp.log(1 + jnp.exp(tmp))

# this works!
def tilted_bound_newton_step(mu, var, xi):
    tmp = mu + 0.5 * (1 - 2 * xi) * var
    g = -var * xi + sigmoid(tmp) * var
    H = -var - sigmoid(tmp) * sigmoid(-tmp) * var**2
    xi = xi - g / H
    return xi

# for fori_loop
def tilted_bound_newton_step2(i, val):
    mu = val['mu']
    var = val['var']
    xi = val['xi']

    xi = tilted_bound_newton_step(mu, var, xi)
    return dict(mu=mu, var=var, xi=xi)

def newton_xi(mu, var):
    val_init = dict(mu=mu, var=var, xi=0.1)
    val = fori_loop(0, 100, tilted_bound_newton_step2, val_init)
    return val['xi']

mu = -10
var = 20
xi = 0.5
print(f'bound = {tilted_bound(mu, var, xi)}, xi = {xi}')
for _ in range(10):
    xi = tilted_bound_newton_step(mu, var, xi)
    print(f'bound = {tilted_bound(mu, var, xi)}, xi = {xi}')

print(newton_xi(-10, 100000))

#%%
# the elbo goes to infinity for large variance
# lets see where that is happening
import jax.numpy as jnp
from tilted import tilted_psi
from utils import *
import matplotlib.pyplot as plt
from utils import plot_contour
from ser import initialize_re

tau0 = 10
mu_range = (1, 3, 50)
var_range = (0.0001, 5, 50)
delta_range = (-2, 2, 50)
mu2 = 1

re = initialize_re(n, mu2=mu2)
params = dict(mu=2, var=0.00001, delta=-1)
params2 = dict(tau0=tau0, xi=np.ones(n)*0.5)
params2['xi'] = xi_fixed_point(data, re, params, params2)
params2['xi'] = xi_fixed_point(data, re, params, params2)
params2['xi'] = xi_fixed_point(data, re, params, params2)
params2['xi'] = xi_fixed_point(data, re, params, params2)
print(tilted_elbo(data, re, params, params2))

#%%
def tilted_obj(x, data, re, params2):
    params = param2dict(x)
    return tilted_elbo(data, re, params, params2)

obj = lambda x: -1 * tilted_obj(x, data, re, dict(tau0=1, xi=np.ones(n) * 0.5))

from jax.scipy.optimize import minimize
fit = minimize(obj, np.array([0., 0., 0.]), method='bfgs', options=dict(maxiter = 10))

#%%
# fit ser
data = simulate_data(n, p)
params = initialize_ser_params(n, p, 1)
re = initialize_re(n, mu2=0)
ser = fit_tilted_ser(data, re, params, {})
ser['summary']['lbf']

#%%
# fit ibss
ibss_fit = ibss2m_tilted(data['X'], data['y'], L=3)
ibss_fit2 = ibss2m_tilted(data['X'], data['y'], L=3, keep_2m=False, tol=1e-6)


# %%
