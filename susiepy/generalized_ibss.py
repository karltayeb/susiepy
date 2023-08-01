import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
import jax.numpy as jnp
import jax
from jax import jit
from susiepy.logistic_regression_newton import logistic_regression_functions
import time
import inspect
from jaxopt import LBFGS
from typing import Callable, Any
from numpy.typing import NDArray

def extract_args(fun: Callable, arguments: dict) -> dict:
    """Extract arguments of a function from a dictionary of arguments

    Args:
        fun (Callable): a function
        arguments (dict): a dictionary

    Returns:
        dict: a dictionary composed of elements of arguments which are arguments to fun
    """
    params = inspect.signature(fun).parameters.keys()
    subargs = {p: arguments[p] for p in params}
    return subargs
 
def exec(fun: Callable, arguments: dict) -> Any:
    """Execute a function, passing `arguments` as kwargs
    note that `arguments` may contain other elements 
    that are not arguments to `fun

    Args:
        fun (Callable): a function to call
        arguments (dict): a dictionary containing keyword arguments 
            (and possibly other data to be disregarded)

    Returns:
        Any: the output of fun called with arguments
    """
    return fun(**extract_args(fun, arguments))

def extract(res, key):
    L = len(res['ser_fits'])
    out = np.array([res['ser_fits'][l][key] for l in range(L)])
    return out

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

@jit
def optimize_prior_variance(pi, betahat, shat2, ll, ll0):
    fun = lambda ln_prior_variance: -1. * compute_lbf_ser(jnp.exp(ln_prior_variance), pi, betahat, shat2, ll, ll0)
    fun_vg = jax.value_and_grad(fun)
    solver = LBFGS(fun=fun_vg, value_and_grad = True)
    opt = solver.run(np.array(0.))
    return jnp.exp(opt.params)

def gibss_generator(regression_functions: dict):
    
    # unpack regression_function
    fit_null = regression_functions['fit_null']
    fit_vmap = regression_functions['fit_vmap_jit']
    fit_init_vmap = regression_functions['fit_from_init_vmap_jit'] 
    
    def fit_ser(X: NDArray, y: NDArray, offset: NDArray = None, weights: NDArray = None, prior_variance: float = 1.0, pi: NDArray = None, estimate_prior_variance: bool = True) -> dict:
        """Fit a single effect regression
        assumes regression function returns MLE and standard error
        uses an aymptotic approximation for the posterior effects, 
        and Laplace approximation for the Bayes factors

        Args:
            X (NDArray): n x p matrix of covaraites
            y (NDArray): n vector of responses
            offset (NDArray, optional): n vector of offset values. Defaults to None.
            weights (NDArray, optional): n vector of observation weights. Defaults to None.
            prior_variance (float, optional): prior variance. Defaults to 1.0.
            pi (NDArray, optional): p vector, probability of selecting each of p variables, sums to 1. Defaults to None.
            estimate_prior_variance (bool, optional): estimate the prior variance by optimizing the approximate BF. Defaults to True.

        Returns:
            dict: a dictionary summarizing the SER fit
        """
        n, p = X.shape
        
        if offset is None:
            offset = 0.
        if weights is None:
            weights = 1. 
        if pi is None:
            pi = np.ones(p) / p
        
        tic = time.perf_counter() # start timer
        
        # fit null model
        ll0_fit = fit_null(y, offset, weights, 100)
        ll0 = ll0_fit['ll']
        null_intercept = ll0_fit['coef'][0]
        
        # fit univariate regression for each variable
        penalty = 1e-5
        maxiter = 50
        fit_mle = fit_vmap(X, y, offset, weights, penalty, maxiter)
        
        # unpack results
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
        
        # asymptotic approximation of posterior distribution
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
            #glm_fit = fit_mle,
            ll = np.array(fit_mle['ll']),
            ll0 = np.array(ll0),
            # prior
            prior_variance = prior_variance,
            pi = pi,
            #tracking
            elapsed_time = toc - tic
        )
        
        # record predictions
        res['psi'] = X @ (res['post_mean'] * res['alpha']) + jnp.inner(res['intercept'], res['alpha'])
        return res
        
    def generalized_ibss(X: NDArray, y: NDArray, L: int, estimate_prior_variance: bool =True, maxit: int = 20, tol: float = 1e-6):
        """Fit generalized IBSS
        Apply IBSS, using `fit_ser` in the inner loop

        Args:
            X (NDArray): n x p matrix of covaraites
            y (NDArray): n vector of responses
            L (int): number of non-zero effects
            estimate_prior_variance (bool, optional): estimate the prior variance by optimizing the approximate BF. Defaults to True.
            maxit (int, optional): maximum number of iterations of IBSS. Defaults to 20.
            tol (float, optional): tolerance for sum of squared difference in prediction to declare convergence. Defaults to 1e-6.
        """
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
        
        # extract results from sers into np.array
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

    return fit_ser, generalized_ibss


logistic_ser, logistic_gibss = gibss_generator(logistic_regression_functions)

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
    

        