# Code for taking a log_likelihood function and performing Newton-Raphson
# this implementation starts with a stepsize of 1
# and halves the stepsize if the likelihood did not increase
# for convex log-likelihood this will converge to global optimum
import jax
from jax import jit
import jax.numpy as jnp


    
def newton_raphson_generator(log_likelihood, coef_initializer):
    
    # compute gradient and hessian
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
        coef_init = coef_initializer(x, y, offset, weights, penalty)
        state_init = make_state(coef_init, x, y, offset, weights, penalty, 1., maxiter)
        return state_init
    
    # vectorize state initializer over `x`
    make_init_state_vmap = jax.vmap(make_init_state, (1, None, None, None, None, None))
    make_init_state_vmap_jit = jax.jit(make_init_state_vmap)

    # newton step with halving step size
    # if
    @jit
    def lr_newton_step_with_stepsize(state, x, y, offset, weights, penalty):
        """
        Newton Raphston with halving stepsize
        propose a newton step with current stepsize and either:
        1. accept if log likelihood improvies, and reset stepsize to 1
        2. reject if log likelihood decreases, keep current state but halve stepsize
        
        Args:
            state (dict): a dictionary containing the optimization state
            x (Array): 1d vector of covariates
            y (Array): 1d response vector
            offset (Array): 1d vector of fixed offsets
            weights (Array): 1d vector of observation weights
            penalty (Array): l2 penalty for the coefficients
        """
        coef = state['coef'] - state['stepsize'] * jnp.linalg.solve(state['hess'], state['grad'])
        state_new = make_state(coef, x, y, offset, weights, penalty, 1., state['maxiter'])

        ll_decreased = state_new['ll'] < state['ll']
        converged = jnp.abs(state['ll'] - state_new['ll']) < 1e-10

        # construct new state
        # select proposed state if log likelihood did not decrease
        # select old state if log likelihood decreases
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

    def compute_std(H):
        """Compute standard error of coefficients

        Args:
            H (Array): Hessian matrix

        Returns:
            Array: vector of standard errors for coefficients
        """
        return jnp.sqrt(jnp.diag(jnp.linalg.inv(-H)))

    @jit
    def fit_1d(x, y, offset, weights, penalty, maxiter):
        init = make_init_state(x, y, offset, weights, penalty, maxiter)
        cond_fun = lambda state: jax.lax.select(state['converged'] + (state['iter'] >= state['maxiter']), False, True)
        body_fun = lambda state: lr_newton_step_with_stepsize(state, x, y, offset, weights, penalty)
        state = jax.lax.while_loop(cond_fun, body_fun, init)
        state['std'] = compute_std(state['hess'])
        return state

    # vectorize
    fit_vmap = jax.vmap(fit_1d, (1, None, None, None, None, None))
    fit_vmap_jit = jax.jit(fit_vmap)

    def fit_null(y, offset, weights, maxiter):
        x = jnp.zeros(y.size)
        penalty = 1e20 # large l2 penalty-- essentially fit intercept-only model
        null_fit = fit_1d(x, y, offset, weights, penalty, maxiter)
        return null_fit
        
    # driver but takes coef initialization
    @jit
    def fit_from_init_1d(coef, x, y, offset, weights, penalty, maxiter):
        state = make_state(coef, x, y, offset, weights, penalty, 1.0, maxiter)
        cond_fun = lambda state: jax.lax.select(state['converged'] + (state['iter'] >= state['maxiter']), False, True)
        body_fun = lambda state: lr_newton_step_with_stepsize(state, x, y, offset, weights, penalty)
        state = jax.lax.while_loop(cond_fun, body_fun, state)
        return state

    # vectorize
    fit_from_init_vmap = jax.vmap(fit_from_init_1d, (0, 1, None, None, None, None, None))
    fit_from_init_vmap_jit = jax.jit(fit_from_init_vmap)
   
    regression_functions = dict(
        ll_grad = ll_grad,
        ll_hess = ll_hess,
        make_state = make_state,
        make_init_state = make_init_state,
        make_init_state_vmap_jit = make_init_state_vmap_jit,
        lr_newton_step_with_stepsize = lr_newton_step_with_stepsize,
        compute_std = compute_std,
        fit_1d = fit_1d,
        fit_null = fit_null,
        fit_vmap_jit = fit_vmap_jit,
        fit_from_init_1d = fit_from_init_1d,
        fit_from_init_vmap_jit = fit_from_init_vmap_jit  
    )
    
    return regression_functions
