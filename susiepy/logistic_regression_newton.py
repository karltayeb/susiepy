import jax.numpy as jnp
import jaxopt
from jax import value_and_grad, vmap, jit
import jax

def sigmoid(r):
    return 1 / (1 + jnp.exp(-r))

def logodds(p):
    p = jnp.clip(p, 1e-14, 1-1e-14)
    return jnp.log(p) - jnp.log(1-p)

def predict(beta0, beta, x, offset):
    return (x * beta) + beta0 + offset

def log_likelihood(coef, x, y, offset, obs_weights, penalty):
    beta0 = coef[0]
    beta = coef[1]
    psi = predict(beta0, beta, x, offset)
    psi = jnp.clip(psi, -100, 100) # bound the log-odds
    ll = y * psi - jnp.log1p(jnp.exp(psi))
    return(jnp.sum(ll * obs_weights) + penalty * beta**2) # weighted log likelihood, l2 penalty

ll_grad = jax.jit(jax.grad(log_likelihood))
ll_hess = jax.jit(jax.hessian(log_likelihood))

# state of optimization-- keep track of parameters, iterations, convergence, stepsize, etc.
@jit
def make_state(coef, x, y, offset, weights, penalty, stepsize, maxiter):
    g = ll_grad(coef, x, y, offset, weights, penalty)
    H = ll_hess(coef, x, y, offset, weights, penalty)
    ll = log_likelihood(coef, x, y, offset, weights, penalty)
    state = dict(coef=coef, grad=g, hess=H, ll=ll, stepsize=stepsize, converged=False, iter=0, maxiter=maxiter)
    return(state)

# state of optimization at initialization
@jit
def make_init_state(x, y, offset, weights, penalty, maxiter):
    coef_init = jnp.hstack([logodds(jnp.mean(y)) - jnp.mean(offset), 0.])
    state_init = make_state(coef_init, x, y, offset, weights, penalty, 1., maxiter)
    return state_init
    
make_init_state_vmap = vmap(make_init_state, (1, None, None, None, None, None))
make_init_state_vmap_jit = jit(make_init_state_vmap)

# basic newton step-- NOT USED
@jit
def lr_newton_step(state, x, y, offset, weights, penalty):
    coef = state['coef'] - jnp.linalg.solve(state['hess'], state['grad'])
    state = make_state(coef, x, y, offset, weights, penalty, 1., state['maxiter'])
    return(state)

# decrease stepsize and keep prior iterate it log likelihood does not increase
# finds *an* increasing step, but weaker than a line search for finding optimal step
# but easier to implement with jittable jax -- WE USE THIS
@jit
def lr_newton_step_with_stepsize(state, x, y, offset, weights, penalty):
    coef = state['coef'] - state['stepsize'] * jnp.linalg.solve(state['hess'], state['grad'])
    state_new = make_state(coef, x, y, offset, weights, penalty, 1., state['maxiter'])

    ll_decreased = state_new['ll'] < state['ll']
    converged = jnp.sum((state['coef'] - state_new['coef'])**2) < 1e-6

    # using jax.lax.select to construct state instead of
    # if ll_decreased:
    #   state_final = state
    # else:
    #   state_final = state_new
    state_final = dict(
        coef = jax.lax.select(ll_decreased, state['coef'], state_new['coef']),
        grad = jax.lax.select(ll_decreased, state['grad'], state_new['grad']),
        hess = jax.lax.select(ll_decreased, state['hess'], state_new['hess']),
        ll = jax.lax.select(ll_decreased, state['ll'], state_new['ll']),
        stepsize = jax.lax.select(ll_decreased, state['stepsize'] / 2., 1.),
        converged = converged,
        iter = state['iter'] + 1,
        maxiter = state['maxiter']
    )
    return(state_final)

# driver function for fitting univaraite logistic regression
@jit
def fit_logistic_regression(x, y, offset, weights, penalty, maxiter):
    init = make_init_state(x, y, offset, weights, penalty, maxiter)
    cond_fun = lambda state: jax.lax.select(state['converged'] + (state['iter'] >= state['maxiter']), False, True)
    body_fun = lambda state: lr_newton_step_with_stepsize(state, x, y, offset, weights, penalty)
    state = jax.lax.while_loop(cond_fun, body_fun, init)
    return state

# vectorize
fit_logistic_regression_vmap = vmap(fit_logistic_regression, (1, None, None, None, None, None))
fit_logistic_regression_vmap_jit = jit(fit_logistic_regression_vmap)

def example():
    import numpy as np
    beta0 = -2
    beta = 1
    n = 7500
    p = 4500
    x = np.random.normal(0, 1, n)
    logits = beta0 + beta * x
    y = np.random.binomial(1, sigmoid(logits), size=logits.size)
    
    beta0_init = logodds(jnp.mean(y)) 
    beta_init = 0.
    init = jnp.hstack([beta0_init, beta_init])
    
    offset = np.zeros_like(y)
    weights = np.ones_like(y)
    penalty = 0.
    
    res1 = fit_logistic_regression(x, y, offset, weights, penalty, maxiter=10)
    
    X = np.random.normal(0, 1, n * p).reshape(n, -1)
    X[:, 0] = x
    # note: apparently you cant use key word arguments-- all argumnets are positivion after vmap 
    res2 = fit_logistic_regression_vmap(X[:, :10], y, offset, weights, penalty, 10)
    res3 = fit_logistic_regression_vmap_jit(X, y, offset, weights, penalty, 10)

