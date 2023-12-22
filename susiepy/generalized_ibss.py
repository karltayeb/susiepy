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

def gibss_generator(fit_ser: Callable):
        
    def generalized_ibss(X: NDArray, y: NDArray, L: int, estimate_prior_variance: bool =True, maxit: int = 20, tol: float = 1e-6, n_chunks: int = 1, penalty: float = 1e-5, maxit_inner: int = 50,):
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
            ser_fits[l] = fit_ser(X, y, psi, estimate_prior_variance=estimate_prior_variance, n_chunks = n_chunks, penalty = penalty, maxiter = maxit_inner)
            psi = psi + ser_fits[l]['psi']
        res = dict(ser_fits = ser_fits, iter = 0)

        for i in range(1, maxit):
            psi_old = psi
            for l in range(L):
                psi = psi - ser_fits[l]['psi']
                ser_fits[l] = fit_ser(X, y, psi, estimate_prior_variance=estimate_prior_variance, n_chunks = n_chunks, penalty = penalty)
                psi = psi + ser_fits[l]['psi']
            
            res = dict(ser_fits = ser_fits, iter = i)
            
            # check convergence of predictions
            diff = np.max((psi - psi_old)**2)
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

    return generalized_ibss

 