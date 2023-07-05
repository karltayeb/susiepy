import jax.numpy as jnp
import numpy as np
from jax import grad, hessian
from jax.lax import while_loop
from jax import jit, vmap
from jax.scipy.special import logsumexp
from susiepy.utils import sigmoid, normal_kl, logit
from jax.scipy.optimize import minimize
from jaxopt import Bisection

def tilted_psi(data: dict, re: dict, params: dict):
    psi0 = re['mu']
    Vpsi0 = re['mu2'] - re['mu']**2

    psi = params['delta'] + data['X'] * params['mu']
    Vpsi = data['X']**2 * params['var']

    return psi + psi0, Vpsi + Vpsi0

def tilted_bound(data: dict, re: dict, params: dict, params2: dict):
    psi, Vpsi = tilted_psi(data, re, params)
    tmp = (psi + 0.5 * (1 - 2 * params2['xi']) * Vpsi)
    bound = data['y'] * psi \
        - 0.5 * params2['xi']**2 * Vpsi \
        + jnp.log(sigmoid(-tmp))
        #- jnp.log(1 + jnp.exp(tmp))
    bound = jnp.sum(bound)
    return bound

def xi_body_fun(val):
    # unpack
    old_xi = val['xi']
    psi = val['psi']
    Vpsi = val['Vpsi']
    tol = val['tol']

    # update xi
    tmp = psi - 0.5 * (1 - 2 * old_xi) * Vpsi
    xi = sigmoid(tmp)

    # check convergence
    diff = jnp.abs(xi - old_xi).sum()
    val = dict(xi=xi, psi=psi, Vpsi=Vpsi, diff=diff, tol=tol)
    return val

def xi_fixed_point(data: dict, re: dict, params: dict, params2: dict):
    psi, Vpsi = tilted_psi(data, re, params)
    xi = params2['xi']
    val_init = dict(xi=xi, psi=psi, Vpsi=Vpsi, diff=100, tol=1e-3)
    val = while_loop(cond_fun, xi_body_fun, val_init)
    xi = val['xi']
    return xi

def xi_optimiality_fun(xi, mean, var):
    phi = mean + var/2 * (1 - 2 * xi)
    return xi - sigmoid(phi)

@jit
def update_xi(mean, var):
    bisec = Bisection(optimality_fun=xi_optimiality_fun, lower=1e-10, upper=1-1e-10, jit=True)
    return bisec.run(mean=mean, var=var).params


Bisection(xi_optimiality_fun, 1e-10, 1-1e-10, 30,)
def tilted_elbo(data: dict, re: dict, params: dict, params2: dict):
    elbo = tilted_bound(data, re, params, params2) \
        - normal_kl(params['mu'], params['var'], 0, 1/params2['tau0'])
    return elbo

# convert parameter dictionary to unconstrained parameter vector
# maybe constrain mu, delta to interval e.g. [-a, a]

# f maps optimization parameters to model parameters
f = lambda x: x # sigmoid(x) * 20 - 10
# g is the inverse of f
g = lambda y: y #logit((y + 10) / 20)

# map optimization params to parameter dictionary
param2dict = lambda x: dict(
    mu = f(x[0]),
    var=jnp.exp(x[1]),
    delta=f(x[2])
)

# map parameter dictionary to optimization parameters
dict2param = lambda d: jnp.array([
    g(d['mu']),
    jnp.log(d['var']),
    g(d['delta'])
])

# objective function with mu, var, and delta in vector format
def tilted_obj(x, data, re, params2):
    params = param2dict(x)
    return tilted_elbo(data, re, params, params2)

# gradient and hessian for newton method
tilted_grad = grad(tilted_obj)
tilted_H = hessian(tilted_obj)

# check convergence by difference
def cond_fun(val):
    not_converged = jnp.abs(val['diff']) > val['tol']
    return not_converged

# one newton step
def body_fun(val):
    # unpack val
    params = val['params']  # params to optimize with newton step
    params2 = val['params2']  # everythin else (e.g. xi)
    data = val['data']
    re = val['re']
    old_elbo = val['elbo']
    tol = val['tol']
    it = val['it']
    maxit = val['maxit']

    # newton step 
    newtonDirection = - jnp.linalg.solve(
        tilted_H(params, data, re, params2),
        tilted_grad(params, data, re, params2)
    )

    # TODO: line search for step size? e.g.
    # params = search_direction(params, newtonDirection, elbo_fun)
    params = params - newtonDirection

    # update variational parameters
    params2['xi'] = xi_fixed_point(
        data, re, param2dict(params), params2)

    # compute elbo
    elbo = tilted_obj(params, data, re, params2)
    diff = elbo - old_elbo

    # print(f'elbo {elbo}, params: {params}')
    val = dict(
        params = params,
        params2 = params2,
        data = data,
        re = re,
        diff = diff,
        elbo = elbo,
        tol = tol,
        it = it + 1,
        maxit = maxit
    )
    return val

def tilted_univariate_lr(data: dict, re: dict, params: dict) -> dict:
    """Fit univariate logistic regression with tilted approximation

    Args:
        data (dict): dictionary of data with keys X, y
        re (dict): "random effect" 
        params (dict): model parameters to be estimated 

    Returns:
        dict: a dictionary with posterior approximation and summary inforamtion
    """
    # break up params into optimization groups
    params2 = params
    params = dict(
        mu = params2.pop('mu'),
        var = 1/params2.pop('tau'),
        delta = params2.pop('delta'),
    )
    params = dict2param(params)

    # initialize
    val_init = dict(
        params = params,
        params2 = params2,
        data = data,
        re = re,
        diff = 1000,
        elbo = tilted_obj(params, data, re, params2),
        tol = 1e-4,
        it = 0,
        maxit = 50
    )

    # fit
    def while_loop2(cond_fun, body_fun, init_val):
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val

    val = while_loop(cond_fun, body_fun, val_init)
    val['params'] = param2dict(val['params'])

    params = dict(
        mu = val['params']['mu'],
        tau = 1/val['params']['var'],
        xi = val['params2']['xi'],
        delta = val['params']['delta'],
        tau0 = val['params2']['tau0']
    )
    track = dict(
        elbo = val['elbo'],
        kl = normal_kl(params['mu'], 1/params['tau'], 0, 1/params2['tau0']),
        diff = val['diff'],
        it = val['it'] 
    )
    res = dict(params=params, track=track)
    return res
