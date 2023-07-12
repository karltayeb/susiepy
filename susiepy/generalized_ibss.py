import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
import jax.numpy as jnp
import jax
from susiepy.logistic_regression_newton import fit_logistic_regression, fit_logistic_regression_vmap_jit
import time
import inspect
from jaxopt import LBFGS

@jax.jit
def compute_lbf_variable(betahat, shat2, ll, ll0, prior_variance):
    # compute corrected log Laplace apprximate BFs
    log_abf = norm.logpdf(betahat, 0, jnp.sqrt(shat2 + prior_variance)) \
        - norm.logpdf(betahat, 0, jnp.sqrt(shat2))
    log_alr_mle = norm.logpdf(betahat, betahat, jnp.sqrt(shat2)) \
        - norm.logpdf(betahat, 0, jnp.sqrt(shat2))
    log_lr_mle = ll - ll0
    log_labf = log_abf - log_alr_mle + log_lr_mle
    return log_labf

@jax.jit
def compute_lbf_ser(prior_variance, pi, betahat, shat2, ll, ll0):
    lbf = compute_lbf_variable(betahat, shat2, ll, ll0, prior_variance) 
    return logsumexp(lbf + jnp.log(pi))

def extract_args(fun, arguments):
    params = inspect.signature(fun).parameters.keys()
    subargs = {p: arguments[p] for p in params}
    return subargs
 
def exec(fun, arguments):
    return fun(**extract_args(fun, arguments))

@jax.jit
def optimize_prior_variance(pi, betahat, shat2, ll, ll0):
    fun = lambda ln_prior_variance: -1. * compute_lbf_ser(jnp.exp(ln_prior_variance), pi, betahat, shat2, ll, ll0)
    fun_vg = jax.value_and_grad(fun)
    solver = LBFGS(fun=fun_vg, value_and_grad = True)
    opt = solver.run(np.array(0.))
    return jnp.exp(opt.params)
    
def fit_ser(X, y, offset = None, weights = None, prior_variance = 1.0, pi=None, estimate_prior_variance=True):
    n, p = X.shape
    
    if offset is None:
        offset = np.zeros(n)
    if weights is None:
        weights = np.ones(n)
    if pi is None:
        pi = np.ones(p) / p
    
    tic = time.perf_counter() # start timer
    
    # fit null model, here its just a univariate regression with extremely large l2 penalty
    ll0_fit = fit_logistic_regression(X[:,1], y, offset, weights, 1e20, 100)
    ll0 = ll0_fit['ll']
    
    # fit univariate regression for each variable
    penalty = 0.
    maxiter = 50
    fit_mle = fit_logistic_regression_vmap_jit(X, y, offset, weights, penalty, maxiter)
    intercept = fit_mle['coef'][:, 0]
    betahat = fit_mle['coef'][:, 1]
    shat2 = fit_mle['std'][:, 1]**2
    ll = fit_mle['ll']
    
    # optimize prior variance
    
    if estimate_prior_variance:
        prior_variance = optimize_prior_variance(pi, betahat, shat2, ll, ll0)
    
    # compute corrected log Laplace apprximate BFs
    log_labf = compute_lbf_variable(betahat, shat2, ll, ll0, prior_variance)
    log_abf = norm.logpdf(betahat, 0, np.sqrt(shat2 + prior_variance)) \
        - norm.logpdf(betahat, 0, np.sqrt(shat2))
    log_alr_mle = norm.logpdf(betahat, betahat, np.sqrt(shat2)) \
        - norm.logpdf(betahat, 0, np.sqrt(shat2))
    log_lr_mle = fit_mle['ll'] - ll0
    log_labf = log_abf - log_alr_mle + log_lr_mle
    
    # compute pips
    log_alpha_unnormalized = log_labf + np.log(pi)
    alpha = np.exp(log_alpha_unnormalized - logsumexp(log_alpha_unnormalized))
    alpha = alpha / alpha.sum() # sometimes logsumexp isnt that accurate
    
    #
    post_precision = 1 / prior_variance + 1 / shat2
    post_variance = 1 / post_precision
    post_mean = (1 / shat2) * post_variance * betahat
    
    toc = time.perf_counter() # start timer

    res = dict(
        # posterior
        post_mean = np.array(post_mean), 
        post_variance = np.array(post_variance),
        alpha = np.array(alpha),
        lbf = np.array(log_labf),
        lbf_ser = logsumexp(log_labf + np.log(pi)),
        # mle
        betahat = np.array(betahat), 
        intercept = np.array(intercept), 
        shat2 = np.array(shat2),
        glm_fit = fit_mle,
        ll = np.array(fit_mle['ll']),
        ll0 = np.array(ll0),
        # prior
        prior_variance = prior_variance,
        pi = pi,
        #tracking
        elapsed_time = toc - tic
    )
    
    # record predictions
    res['psi'] = X @ (res['post_mean'] * res['alpha'])
    return res
    

def extract(res, key):
    L = len(res['ser_fits'])
    out = np.array([res['ser_fits'][l][key] for l in range(L)])
    return out

def generalized_ibss(X, y, L, estimate_prior_variance=True, maxit=20, tol = 1e-6):
    
    tic = time.perf_counter() # start timer
    
    # first iteration
    psi = np.zeros_like(y) # initialize offset E[Xb]
    ser_fits = dict()
    for l in range(L):
        ser_fits[l] = fit_ser(X, y, psi, estimate_prior_variance=estimate_prior_variance)
        psi = psi + ser_fits[l]['psi']
    res = dict(ser_fits = ser_fits, iter = 0)

    for i in range(1, maxit):
        psi_old = psi
        for l in range(L):
            psi = psi - ser_fits[l]['psi']
            ser_fits[l] = fit_ser(X, y, psi, estimate_prior_variance=estimate_prior_variance)
            psi = psi + ser_fits[l]['psi']
        
        res = dict(ser_fits = ser_fits, iter = i)
        
        # check convergence of predictions
        diff = np.sum((psi - psi_old)**2) 
        print(f'iter = {i}, diff = {diff}')
        if diff < tol:
            break
    toc = time.perf_counter() # stop timer
    keys = dict(
        alpha = 'alpha',
        mu = 'post_mean',
        var = 'post_variance',
        betahat = 'betahat',
        intercept = 'intercept',
        shat2 = 'shat2',
        lbf = 'lbf',
        lbf_ser = 'lbf_ser',
        ll = 'll',
        ll0 = 'll0',
        prior_variance = 'prior_variance',
        psi = 'psi')
    summary = {k1: extract(res, k2) for k1, k2 in keys.items()}
    res.update(summary)
    res['elapsed_time'] = toc - tic
    return(res)

def test_fit_ser():
    n = 1000
    p = 100
    X = np.random.normal(size=n*p).reshape(n, -1)
    logit = -2 + X[:, 1]
    y = np.random.binomial(1, 1/(1 + np.exp(-logit)), n)
   
    ser_fit1 = fit_ser(X, y, estimate_prior_variance=False) 
    ser_fit2 = fit_ser(X, y)
    
    ser_fit2['lbf_ser'] - ser_fit1['lbf_ser']
    
    susie_fit = generalized_ibss(X, y, L=5, estimate_prior_variance=False)
    susie_fit2 = generalized_ibss(X, y, L=5, estimate_prior_variance=True)
    

        