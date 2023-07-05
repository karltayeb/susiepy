import jax.numpy as jnp
import jaxopt
from jax import value_and_grad, vmap, jit
import jax
from jaxopt import BacktrackingLineSearch

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

@jit
def make_state(coef, x, y, offset, weights, penalty, stepsize):
    g = ll_grad(coef, x, y, offset, weights, penalty)
    H = ll_hess(coef, x, y, offset, weights, penalty)
    ll = log_likelihood(coef, x, y, offset, weights, penalty)
    state = dict(coef=coef, grad=g, hess=H, ll=ll, stepsize=stepsize, converged=False, iter=0)
    return(state)

@jit
def make_init_state(x, y, offset, weights, penalty):
    coef_init = jnp.hstack([logodds(jnp.mean(y)) - jnp.mean(offset), 0.])
    state_init = make_state(coef_init, x, y, offset, weights, penalty, 1.)
    return state_init
    
make_init_state_vmap = vmap(make_init_state, (1, None, None, None, None))
make_init_state_vmap_jit = jit(make_init_state_vmap)

@jit
def lr_newton_step(state, x, y, offset, weights, penalty):
    coef = state['coef'] - jnp.linalg.solve(state['hess'], state['grad'])
    state = make_state(coef, x, y, offset, weights, penalty, 1.)
    return(state)

lr_newton_step_vmap = jit(vmap(lr_newton_step, ({'coef': 0, 'grad': 0, 'hess': 0, 'll': 0}, 1, None, None, None, None)))

# iterate newton step with jax.lax.fori_loop
@jit
def lr_loop(init, x, y, offset, weights, penalty, niter):
    body_fun = lambda i, state: lr_newton_step(state, x, y, offset, weights, penalty)
    state = jax.lax.fori_loop(0, niter, body_fun, init)
    return(state)

lr_loop_vmap = jit(vmap(lr_loop, ({'coef': 0, 'grad': 0, 'hess': 0, 'll': 0}, 1, None, None, None, None, None)))

# add line search to newton step to get improving step 
@jit
def lr_newton_step_with_linesearch(state, x, y, offset, weights, penalty):    
    descent_direction = -1 * jnp.linalg.solve(state['hess'], state['grad'])
    fun = lambda coef: -1 * log_likelihood(coef, x, y, offset, weights, penalty)
    ls = BacktrackingLineSearch(fun=fun, 
                                maxiter=20, 
                                condition="strong-wolfe",
                                decrease_factor=0.5)
    
    stepsize, ls_state = ls.run(init_stepsize=1.0, 
                                params=state['coef'],
                                descent_direction=descent_direction,
                                value= -1 * state['ll'],
                                grad= -1 * state['grad'])
    
    coef = ls_state.params
    state = make_state(coef, x, y, offset, weights, penalty, 1.)
    return(state)

lr_newton_step_with_linesearch_vmap = jit(vmap(lr_newton_step_with_linesearch, ({'coef': 0, 'grad': 0, 'hess': 0, 'll': 0}, 1, None, None, None, None)))

# Perform naive newton step, perform line search for variables which ll decreased
def newton_step2(states, X, y, offset, weights, penalty):
    # newtwon step with size 1
    states2 = lr_newton_step_vmap(states, X, y, offset, weights, penalty)
    
    # check if likelihood decreased
    linesearch_idx = jnp.where(states2['ll'] < states['ll'])[0]
    subset_states = dict(
        coef = states['coef'][linesearch_idx,],
        grad = states['grad'][linesearch_idx,],
        hess = states['hess'][linesearch_idx,],
        ll = states['ll'][linesearch_idx]
    )
    linesearch_states = lr_newton_step_with_linesearch_vmap(
        subset_states, X[:, linesearch_idx], y, offset, weights, penalty)
    
    states3 = dict(
        coef = states2['coef'].at[linesearch_idx,:].set(linesearch_states['coef']),
        grad = states2['grad'].at[linesearch_idx,:].set(linesearch_states['grad']),
        hess = states2['hess'].at[linesearch_idx,:].set(linesearch_states['hess']),
        ll = states2['ll'].at[linesearch_idx,].set(linesearch_states['ll'])
    )
    return states3

@jit
def lr_newton_step_with_stepsize(state, x, y, offset, weights, penalty):
    coef = state['coef'] - state['stepsize'] * jnp.linalg.solve(state['hess'], state['grad'])
    state_new = make_state(coef, x, y, offset, weights, penalty, 1.)

    ll_decreased = state_new['ll'] < state['ll']
    converged = jnp.sum((state['coef'] - state_new['coef'])**2) < 1e-6
    state_final = dict(
        coef = jax.lax.select(ll_decreased, state['coef'], state_new['coef']),
        grad = jax.lax.select(ll_decreased, state['grad'], state_new['grad']),
        hess = jax.lax.select(ll_decreased, state['hess'], state_new['hess']),
        ll = jax.lax.select(ll_decreased, state['ll'], state_new['ll']),
        stepsize = jax.lax.select(ll_decreased, state['stepsize'] / 2., 1.),
        converged = converged,
        iter = state['iter'] + 1
    )
    return(state_final)


state = make_init_state(x, y, offset, weights, penalty)
while(not state['converged']):
    state = lr_newton_step_with_stepsize(state, x, y, offset, weights, penalty)
    print(f'coef = {state["coef"]}, ll = {state["ll"]}, converged = {state["converged"]}')

@jit
def fit_logistic_regression(x, y, offset, weights, penalty):
     init = make_init_state(x, y, offset, weights, penalty)
     cond_fun = lambda state: jax.lax.select(state['converged'], False, True)
     body_fun = lambda state: lr_newton_step_with_stepsize(state, x, y, offset, weights, penalty)
     state = jax.lax.while_loop(cond_fun, body_fun, init)
     return state

fit_logistic_regression_vmap = vmap(fit_logistic_regression, (1, None, None, None, None))
fit_logistic_regression_vmap_jit = jit(fit_logistic_regression_vmap)

res = fit_logistic_regression_vmap_jit(X, y, offset, weights, penalty)
res['converged']
@jit
def fit_logistic_regression(x, y, offset, weights, penalty, niter):
    coef_init = jnp.hstack([logodds(jnp.mean(y)) - jnp.mean(offset), 0.])
    state_init = make_state(coef_init, x, y, offset, weights, penalty)
    state = lr_loop(state_init, x, y, offset, weights, penalty, niter)
    return(state)

fit_logistic_regression_vmap = jit(vmap(fit_logistic_regression, (1, None, None, None, None, None)))

@jit
def fit_logistic_regression2(init, x, y, offset, weights, penalty, niter):
    coef_init = jnp.hstack([logodds(jnp.mean(y)) - jnp.mean(offset), 0.])
    state_init = make_state(coef_init, x, y, offset, weights, penalty)
    state = lr_loop(state_init, x, y, offset, weights, penalty, niter)
    return(state)


@jit
def objective(coefs, x, y, offset, obs_weight):
    beta0 = coefs[0]
    beta = coefs[1]
    return -1 * log_likelihood(beta0, beta, x, y, offset, obs_weight).astype(float)

# useing jax transformations instead of computing explicitly
objective_and_grad = jit(value_and_grad(objective))


def fit_bfgs(init_params, x, y, offset, obs_weight):
    solver = jaxopt.BFGS(fun=objective_and_grad, value_and_grad = True, maxiter=100)
    res = solver.run(init_params, x=x, y=y, offset=offset, obs_weight=obs_weight)
    return(res)

# vectorize over init and x
fit_bfgs_vectorized = vmap(fit_bfgs, (0, 1, None, None, None))

# jit for fast computation
fit_bfgs_vectorized_jit = jit(fit_bfgs_vectorized)


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
    
    penalty = 1e-10
    ll0 = log_likelihood(init, x, y, offset, weights, penalty)
    par = lr_newton_step(init, x, y, offset, weights, penalty)
    ll1 = log_likelihood(par, x, y, offset, weights, penalty)
    print(f'll = {ll0:.3f}, coef = {init}\nll = {ll1:.3f}, coef={par}')
    
    #%time 
    res = fit_bfgs(init, x, y, offset, weights)
    
    first_jit_fit = fit_jit(init, x, y, offset, weights)
    res = fit_bfgs_jit(init, x, y, offset, weights)
    
    Init = jnp.vstack([init for _ in range(p)])
    X = np.random.normal(0, 1, n * p).reshape(n, -1)
    X[:, 0] = x
    res = fit_bfgs_vectorized(Init[:3, :], X[:, :3], y, offset, weights)
    res = fit_bfgs_vectorized_jit(Init, X[:, :], y, o, w)
 
    # alt arguments, figure out what trigures re-comilation
    # shouldn't change as long as input shape doesnt change
    # which is good because for a single SuSiE run we only need to compile once
    Init2 = np.array(Init)
    Init2[:, 1] = np.random.normal(0, 0.01, Init2.shape[0])
    o = np.random.normal(0, 1, offset.size)
    w = np.abs(np.random.normal(0, 1, offset.size))

    params = init
    
    for i in range(1000):
        ll = log_likelihood(*params, x, y, offset, weights)
        q1 = compute_local_quadratic_approximation(params, params, x, y, offset, weights)
        q1_grad = quad_grad(params, params, x, y, offset, weights)
        q1_hess = quad_hessian(params, params, x, y, offset, weights)
        params = params - jnp.linalg.solve(q1_hess, q1_grad)
        if i % 20 == 0:
            print(f'i = {i}, params = {params}, ll={ll}')
    #quad_grad(params2, params, x, y, offset, weights)
    
    params = init
    for i in range(20):
        params = lr_newton_step(params, x, y, offset, weights)
        ll = log_likelihood2(params, x, y, offset, weights)
        print(f'i = {i}, params = {params}, ll={ll}')


def compute_std_err():
    pass
