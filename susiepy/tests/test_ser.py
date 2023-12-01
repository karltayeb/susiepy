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


# %%
