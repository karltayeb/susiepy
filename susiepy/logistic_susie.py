import jax.numpy as jnp
from susiepy.newton_raphson import newton_raphson_generator
from susiepy.generalized_ser import ser_generator
from susiepy.generalized_ibss import gibss_generator

def sigmoid(r):
    return 1 / (1 + jnp.exp(-r))


def logodds(p):
    p = jnp.clip(p, 1e-14, 1-1e-14)
    return jnp.log(p) - jnp.log(1-p)


def predict(beta0, beta, x, offset):
    return (x * beta) + beta0 + offset


def logistic_regression_log_likelihood(coef, x, y, offset, obs_weights, penalty):
    """Compute log likelihood for logistic regression

    Args:
        coef (NDArray): 2 vector with intercept and effect
        x (NDArray): n vector of covariate values
        y (NDArray): n vector of responses
        offset (NDArray): n vector of fixed offsets
        obs_weights (NDArray): n vector of observation weights
        penalty (float): l2 penalty for effect
    """
    beta0 = coef[0]
    beta = coef[1]
    psi = predict(beta0, beta, x, offset)
    psi = jnp.clip(psi, -100, 100) # bound the log-odds
    ll = y * psi - jnp.log1p(jnp.exp(psi))
    return(jnp.sum(ll * obs_weights) - penalty * beta**2) # weighted log likelihood, l2 penalty


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
    
logistic_ser = ser_generator(logistic_regression_functions)
logistic_gibss = gibss_generator(logistic_ser)


def example():
    import numpy as np
    from susiepy.logistic_susie import sigmoid, logistic_ser, logistic_gibss, logistic_regression_functions
    beta0 = -2
    beta = 0.5
    n = 50000
    p = 100
    x = np.random.normal(0, 1, n)
    logits = beta0 + beta * x
    y = np.random.binomial(1, sigmoid(logits), size=logits.size).astype(float)
    X = np.random.normal(0, 1, n * p).reshape(n, -1)
    X[:, 0] = x
   
    a = logistic_ser(X, y, n_chunks=10)
    a2 = logistic_ser(X, y, n_chunks=1)
    b = logistic_gibss(X, y, L=5, maxit=3, n_chunks=10, tol=1e-10)
    
    # %time a = logistic_regression_functions['fit_vmap_jit'](X, y, 0., 1., 0., 10.)
    # %time b = logistic_regression_functions['fit_vmap_jit_chunked'](X, y, 0., 1., 0., 10., 10)
    
    # %time a = logistic_ser(X, y, n_chunks=1)
    # %time b = logistic_ser(X, y, n_chunks=10)
    
    Xs = np.array_split(X, 10, axis=1)
    res = [logistic_ser(z, y) for z in Xs]
    
    
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
    


    # note: apparently you cant use key word arguments-- all argumnets are positivion after vmap 
    res2 = logistic_regression_functions['fit_vmap_jit'](X[:, :10], y, offset, weights, penalty, 10)
    res3 = logistic_regression_functions['fit_vmap_jit'](X, y, offset, weights, penalty, 10)
    res4 = logistic_ser(X, y)
    res5 = logistic_gibss(X, y, L=2, maxit=2)
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


def test_fit_ser():
    n = 7500 
    p = 1000
    X = np.random.normal(size=n*p).reshape(n, -1)
    logit = -2 + 3* X[:, 1]
    y = np.random.binomial(1, 1/(1 + np.exp(-logit)), n)
   
    ser_fit1 = logistic_ser(X, y, estimate_prior_variance=False) 
    ser_fit2 = logistic_ser(X, y)
    
    ser_fit2['lbf_ser'] - ser_fit1['lbf_ser']
    
    susie_fit = logistic_gibss(X, y, L=5, estimate_prior_variance=False)
    susie_fit2 = logistic_gibss(X, y, L=5, estimate_prior_variance=True)
    
