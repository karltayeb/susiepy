#%%
from susiepy.logistic_susie import logistic_ser, sigmoid
from susiepy.logistic_susie import logistic_gibss 
import numpy as np

def simulate():
    beta0 = -2
    beta = 0.5
    n = 5000
    p = 100
    x = np.random.normal(0, 1, n)
    logits = beta0 + beta * x
    y = np.random.binomial(1, sigmoid(logits), size=logits.size).astype(float)
    X = np.random.normal(0, 1, n * p).reshape(n, -1)
    X[:, 0] = x
    return X, y

def test_ser():
    X, y = simulate()
    fit = logistic_ser(X, y)
    return fit
 
fit = test_ser()
fit['lbf'][0]

# %%
def test_lbf_laplace_mle():
    from susiepy.generalized_ser import compute_lbf_laplace_mle
    from susiepy.generalized_ser import compute_lbf_laplace_mle2
    
    betahat = 1
    shat2 = 0.1
    ll = 10
    ll0 = 0
    prior_variance = 0.123
    
    a = compute_lbf_laplace_mle(betahat, shat2, ll, ll0, prior_variance)
    b = compute_lbf_laplace_mle2(betahat, shat2, ll, ll0, prior_variance)
    assert(a == b)

test_lbf_laplace_mle()

# %%
def test_gibss():
    X, y = simulate()
    fit = logistic_gibss(X, y, 3, True)
    return fit
 
fit = test_gibss()
fit['prior_variance']

# %%
from susiepy.generalized_ser import make_mle_quadrature_rule
from susiepy.logistic_susie import logistic_regression_log_likelihood
from susiepy.logistic_susie import logistic_regression_functions
q16, q16v = make_mle_quadrature_rule(16)
X, y = simulate()
fit = logistic_ser(X, y)
lbf_quad = q16v(X, y, np.zeros_like(y), np.ones_like(y), fit, fit['prior_variance']) - fit['ll0']

# fit_mle = logistic_regression_functions['fit_1d']
# x = X[:, 0]
# # x, y, offset, weights, penalty, maxiter
# mle = fit_mle(x, y, 0., 1., 0., 10)
# mle['betahat'] = mle['coef'][1]
# mle['intercept'] = mle['coef'][0]
# q16(x, y, 0., 1., 0.3, mle) - mle['ll0']
# %%
