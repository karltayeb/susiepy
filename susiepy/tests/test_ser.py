#%%
from susiepy.logistic_susie import logistic_ser, sigmoid
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
    


    
# %%
