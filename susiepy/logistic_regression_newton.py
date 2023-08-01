import jax.numpy as jnp
from jax import vmap, jit
import jax
from susiepy.newton_raphson import newton_raphson_generator

def sigmoid(r):
    return 1 / (1 + jnp.exp(-r))


def logodds(p):
    p = jnp.clip(p, 1e-14, 1-1e-14)
    return jnp.log(p) - jnp.log(1-p)


def predict(beta0, beta, x, offset):
    return (x * beta) + beta0 + offset


def logistic_regression_log_likelihood(coef, x, y, offset, obs_weights, penalty):
    beta0 = coef[0]
    beta = coef[1]
    psi = predict(beta0, beta, x, offset)
    psi = jnp.clip(psi, -100, 100) # bound the log-odds
    ll = y * psi - jnp.log1p(jnp.exp(psi))
    return(jnp.sum(ll * obs_weights) + penalty * beta**2) # weighted log likelihood, l2 penalty


def logistic_regression_coef_initializer(x, y, offset, weights, penalty):
    """Initialize coeficients for logistic regression

    Args:
        x (Array): 1d vector of covariates
        y (Array): 1d response vector
        offset (Array): 1d vector of fixed offsets
        weights (Array): 1d vector of observation weights
        penalty (Array): l2 penalty for the coefficients

    Returns:
        Array: vector of initialized coefficients
    """
    return jnp.hstack([logodds(jnp.mean(y)) - jnp.mean(offset), 0.])


logistic_regression_functions = \
    newton_raphson_generator(logistic_regression_log_likelihood, logistic_regression_coef_initializer)
    

def example():
    import numpy as np
    beta0 = -2
    beta = 0.5
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

    init = logistic_regression_functions['make_init_state'](x, y, offset, weights, penalty, maxiter=10)
    states = dict()
    states[0] = init
    for i in range(1, 20):
        states[i] = logistic_regression_functions['lr_newton_step_with_stepsize'](
            states[i-1], x, y, offset, weights, penalty)
        print(states[i]['ll'])
    coef = states[19]['coef']
   
    from sklearn.linear_model import LogisticRegression 
    sklr = LogisticRegression(random_state=0, penalty=None).fit(x[:, None], y)
    sklr_coef = np.hstack([sklr.intercept_, sklr.coef_.flatten()])
    ll_grad = logistic_regression_functions['ll_grad']
    ll_grad(sklr_coef, x, y, offset, weights, penalty)
    
    logistic_regression_log_likelihood(sklr_coef, x, y, offset, weights, penalty)
    logistic_regression_log_likelihood(coef, x, y, offset, weights, penalty)
    
    fit_logistic_regression = logistic_regression_functions['fit_1d']
    res1 = fit_logistic_regression(x, y, offset, weights, penalty, maxiter=10)
    
    X = np.random.normal(0, 1, n * p).reshape(n, -1)
    X[:, 0] = x

    # note: apparently you cant use key word arguments-- all argumnets are positivion after vmap 
    res2 = logistic_regression_functions['fit_vmap_jit'](X[:, :10], y, offset, weights, penalty, 10)
    res3 = logistic_regression_functions['fit_vmap_jit'](X, y, offset, weights, penalty, 10)

    # use NR generator
    # idx = np.argmin(res3['grad'][:, 0])
    # x2 = X[:, idx]
    
    # coef = res3['coef'][idx,]
    # sklr = LogisticRegression(random_state=0, penalty=None).fit(x2[:, None], y)
    # sklr_coef = np.hstack([sklr.intercept_, sklr.coef_.flatten()])
    # ll_grad(sklr_coef, x2, y, offset, weights, penalty)
    # compute_std(ll_hess(sklr_coef, x2, y, offset, weights, penalty))
    # log_likelihood(sklr_coef, x2, y, offset, weights, penalty)
    # ll_grad(coef, x2, y, offset, weights, penalty)
    # compute_std(ll_hess(coef, x2, y, offset, weights, penalty))
    # log_likelihood(coef, x2, y, offset, weights, penalty)