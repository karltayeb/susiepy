import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from susiepy.logistic_regression_newton import fit_logistic_regression, fit_logistic_regression_vmap_jit
import time

def fit_ser(X, y, offset = None, weights = None, prior_variance = 1.0, pi=None):
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
    
    # compute corrected log Laplace apprximate BFs
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
        post_mean = np.array(post_mean), 
        post_variance = np.array(post_variance),
        betahat = np.array(betahat), 
        intercept = np.array(intercept), 
        shat2 = np.array(shat2),
        alpha = np.array(alpha),
        lbf = np.array(log_labf),
        glm_fit = fit_mle,
        ll = np.array(fit_mle['ll']),
        ll0 = np.array(ll0),
        prior_variance = prior_variance,
        elapsed_time = toc - tic
    )
    # record predictions
    res['psi'] = X @ (res['post_mean'] * res['alpha'])
    return res
    

def generalized_ibss(X, y, L, maxit=20, tol = 1e-6):
    psi = np.zeros_like(y) # initialize offset E[Xb]
    
    tic = time.perf_counter() # start timer
    
    # first iteration
    ser_fits = dict()
    for l in range(L):
        ser_fits[l] = fit_ser(X, y, psi, prior_variance=1.0)
        psi = psi + ser_fits[l]['psi']
    res = dict(ser_fits = ser_fits, iter = 0)

    for i in range(1, maxit):
        psi_old = psi
        for l in range(L):
            psi = psi - ser_fits[l]['psi']
            ser_fits[l] = fit_ser(X, y, psi, prior_variance=1.0)
            psi = psi + ser_fits[l]['psi']
        
        res = dict(ser_fits = ser_fits, iter = i)
        
        # check convergence of predictions
        diff = np.sum((psi - psi_old)**2) 
        print(f'iter = {i}, diff = {diff}')
        if diff < tol:
            break
    toc = time.perf_counter() # stop timer
    
    alpha = np.array([res['ser_fits'][l]['alpha'] for l in range(L)])
    mu = np.array([res['ser_fits'][l]['post_mean'] for l in range(L)])

    res['alpha'] = alpha
    res['mu'] = mu
    res['elapsed_time'] = tic - toc
    return res

        