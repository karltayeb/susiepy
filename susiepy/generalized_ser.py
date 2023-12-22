import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
import jax.numpy as jnp
import jax
from jax import jit
import time
import inspect
from jaxopt import LBFGS
from typing import Callable, Any
from numpy.typing import NDArray
from functools import partial
from numpy.polynomial import hermite

@jax.jit
def compute_lbf_laplace_mle2(betahat, shat2, ll, ll0, prior_variance):
    # compute corrected log Laplace apprximate BFs
    log_abf = norm.logpdf(betahat, 0, jnp.sqrt(shat2 + prior_variance)) \
        - norm.logpdf(betahat, 0, jnp.sqrt(shat2))
    log_alr_mle = norm.logpdf(betahat, betahat, jnp.sqrt(shat2)) \
        - norm.logpdf(betahat, 0, jnp.sqrt(shat2))
    log_lr_mle = ll - ll0
    log_labf = log_abf - log_alr_mle + log_lr_mle
    return log_labf

@jax.jit
def compute_lbf_laplace_mle(betahat, shat2, ll, ll0, prior_variance):
    V = shat2 + prior_variance
    return (ll - ll0) + \
        0.5 * jnp.log(shat2/V) - \
        0.5 * betahat**2 / V

@jax.jit
def compute_lbf_ser(prior_variance, pi, betahat, shat2, ll, ll0):
    lbf = compute_lbf_laplace_mle(betahat, shat2, ll, ll0, prior_variance) 
    return logsumexp(lbf + jnp.log(pi))

@jit
def optimize_prior_variance(pi, betahat, shat2, ll, ll0):
    fun = lambda ln_prior_variance: -1. * compute_lbf_ser(jnp.exp(ln_prior_variance), pi, betahat, shat2, ll, ll0)
    fun_vg = jax.value_and_grad(fun)
    solver = LBFGS(fun=fun_vg, value_and_grad = True)
    opt = solver.run(np.array(0.))
    return jnp.exp(opt.params)

def make_mle_quadrature_rule2(log_likelihood, m=16):
    # compute quadrature points
    qx, qw = hermite.hermgauss(m)
    
    # vectorize log likelihood, evaluate at multiple coef
    # args: (coef, x, y, offset, obs_weights, penalty)
    log_likelihood_vec = jax.vmap(log_likelihood, (0, None, None, None, None, None))

    def logp_quad_mle(x, y, offset, obs_weights, prior_variance, mle_fit):
        """Compute marginal likelihood using quadrature rule from MLE

        Args:
            x (_type_): covariate
            y (_type_): respose
            offset (_type_): offset in lienar prediction
            mle_fit (_type_): dictionary with summary of MLE fit
            tau0 (_type_): prior precision

        Returns:
            float: M-point quadrature
        """        
        
        # quadrature using mle
        bhat = mle_fit['betahat']
        b0 = mle_fit['intercept'] 
        tau0 = 1/prior_variance
        tau1 = -mle_fit['hess'][1, 1]

        tau = tau1 + tau0
        mu = tau1/tau * bhat

        def f(b):
            # log p(y, b, b0) - log q(b)
            # for vector of b
            # q(b) is a chosen so that Hermite quadrature rule works well
            
            # make coefficients 
            coef = jnp.array([jnp.ones(m)*b0, b]).T
            ll = log_likelihood_vec(coef, x, y, offset, obs_weights, 0.)
            logpb = 0.5 * jnp.log(tau0 / (2 *jnp.pi)) - 0.5 * tau0 * b**2
            return jnp.sum(ll) + logpb + 0.5 * tau * (b - mu)**2

        # change of variable for quadrature points
        qb = jnp.sqrt(2/tau) * qx + mu
        
        # compute integral on log scale
        logI = logsumexp(f(qb) + jnp.log(qw))
        return 0.5 * jnp.log(2/tau) + logI
   
    # extract only the arguments we need
    extract_quad_data = lambda fits: {k: fits.get(k) for k in ['betahat', 'intercept', 'hess']}
    
    # vectorize the quadrature rule over columns of X
    # x, y, offset, obs_weights, prior_variance, mle_fit
    _logp_quad_mle_vmap = jax.vmap(
        logp_quad_mle,(1, None, None, None, None, {'betahat': 0, 'intercept': 0, 'hess': 0}))
   
    @jit 
    def logp_quad_mle_vmap(X, y, offset, obs_weights, fits, prior_variance):
       fits2 = extract_quad_data(fits)
       return _logp_quad_mle_vmap(X, y, offset, obs_weights, prior_variance, fits2) 
    
    return logp_quad_mle, logp_quad_mle_vmap

def make_mle_quadrature_rule(m=16):
    # compute quadrature points
    qx, qw = hermite.hermgauss(m)

    def logp_quad_mle(x, y, offset, obs_weights, prior_variance, mle_fit):
        """Compute marginal likelihood using quadrature rule from MLE

        Args:
            x (_type_): covariate
            y (_type_): respose
            offset (_type_): offset in lienar prediction
            mle_fit (_type_): dictionary with summary of MLE fit
            tau0 (_type_): prior precision

        Returns:
            float: M-point quadrature
        """
        # quadrature using mle
        bhat = mle_fit['betahat']
        b0 = mle_fit['intercept']
        tau1 = -mle_fit['hess'][1, 1]

        tau0 = 1/prior_variance
        tau = tau1 + tau0
        mu = tau1/tau * bhat

        def log_joint(b):
            psi = b * x + b0 + offset
            psi = jnp.clip(psi, -100, 100) # bound the log-odds
            ll = y * psi - jnp.log1p(jnp.exp(psi))
            logpb = 0.5 * jnp.log(tau0 / (2 *jnp.pi)) - 0.5 * tau0 * b**2
            return jnp.sum(ll * obs_weights) + logpb
        def f(b):
            return log_joint(b) + 0.5 * tau * (b - mu)**2
        fv = jax.vmap(f)

        qb = jnp.sqrt(2/tau) * qx + mu
        I = logsumexp(fv(qb) + jnp.log(qw))
        return 0.5 * jnp.log(2/tau) + I
    
    extract_quad_data = lambda fits: {k: fits.get(k) for k in ['betahat', 'intercept', 'hess']}
    _logp_quad_mle_vmap = jax.vmap(
        logp_quad_mle,(1, None, None, None, None, {'betahat': 0, 'intercept': 0, 'hess': 0}))
   
    @jit 
    def logp_quad_mle_vmap(X, y, offset, obs_weights, fits, prior_variance):
       fits2 = extract_quad_data(fits)
       return _logp_quad_mle_vmap(X, y, offset, obs_weights, prior_variance, fits2) 
    
    return logp_quad_mle, logp_quad_mle_vmap


def ser_generator(regression_functions: dict):
    # unpack regression_function
    fit_null = regression_functions['fit_null']
    fit_vmap = regression_functions['fit_vmap_jit_chunked']
    
    def _fit_ser(X, y, offset, weights, prior_variance, estimate_prior_variance, pi, n_chunks, penalty, maxiter):
        # fit null model
        ll0_fit = fit_null(y, offset, weights, 100)
        ll0 = ll0_fit['ll']
        null_intercept = ll0_fit['coef'][0]
        
        # fit univariate regression for each variable
        mle = fit_vmap(X, y, offset, weights, penalty, maxiter, n_chunks)
        
        # unpack results
        intercept = mle['coef'][:, 0]
        betahat = mle['coef'][:, 1]
        shat2 = -1/mle['hess'][:,1, 1] # note: assume b0 fixed at mle
        ll = mle['ll']
        
        # optimize prior variance
        # WARNING: this is optimizing an asymptotic approximation of the BF for the SER
        # the quality of this approximation will vary as a function of prior variance
        if estimate_prior_variance:
            prior_variance = optimize_prior_variance(pi, betahat, shat2, ll, ll0)
        
        # compute corrected log Laplace apprximate BFs
        lbf = compute_lbf_laplace_mle(betahat, shat2, ll, ll0, prior_variance)
        
        # compute pips
        log_alpha_unnormalized = lbf + jnp.log(pi)
        alpha = jnp.exp(log_alpha_unnormalized - logsumexp(log_alpha_unnormalized))
        alpha = alpha / alpha.sum() # sometimes logsumexp isnt that accurate
        
        # asymptotic approximation of posterior distribution
        post_precision = 1 / prior_variance + 1 / shat2
        post_variance = 1 / post_precision
        post_mean = (1 / shat2) * post_variance * betahat
        
        res = dict(
            # posterior
            post_mean = post_mean, 
            post_variance = post_variance,
            alpha = alpha,
            lbf = lbf,
            lbf_ser = logsumexp(lbf + jnp.log(pi)),
            # mle
            betahat = betahat, 
            intercept = intercept, 
            hess = mle['hess'],
            shat2 = shat2,
            #glm_fit = fit_mle,
            ll = mle['ll'],
            ll0 = ll0,
            # prior
            prior_variance = prior_variance,
            pi = pi
        )
        
        # record predictions
        res['psi'] = X @ (res['post_mean'] * res['alpha']) + jnp.inner(res['intercept'], res['alpha'])
        return res
    
    def _quad_wrapper(X, y, offset, weights, ser):
        """Recompute lbf, post mean, and post variance using quadrature

        Args:
            X (_type_): _description_
            y (_type_): _description_
            offset (_type_): _description_
            weights (_type_): _description_
            ser (_type_): _description_
        """
        pass 
    
    def fit_ser_mle(X: NDArray, y: NDArray, offset: NDArray = None, weights: NDArray = None, prior_variance: float = 1.0, pi: NDArray = None, estimate_prior_variance: bool = True, n_chunks: int = 1, penalty = 1e-5, maxiter=50) -> dict:
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
        res = _fit_ser(X, y, offset, weights, prior_variance, estimate_prior_variance, pi, n_chunks, penalty, maxiter)
        toc = time.perf_counter() # start timer
        res['elapsed_time'] = toc - tic
        return res
        
    def fit_ser_quad(X: NDArray, y: NDArray, offset: NDArray = None, weights: NDArray = None, prior_variance: float = 1.0, pi: NDArray = None, estimate_prior_variance: bool = True, n_chunks: int = 1, penalty = 1e-5, maxiter=50) -> dict:
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
        res = _fit_ser(X, y, offset, weights, prior_variance, estimate_prior_variance, pi, n_chunks, penalty, maxiter)
        res = _quad_wrapper(X, y, offset, weights, res)
        toc = time.perf_counter() # start timer
        res['elapsed_time'] = toc - tic
        return res
    
    return fit_ser_mle