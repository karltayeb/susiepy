# `susiepy`

## Details

### Vectorized and JIT-ed univariate regression

We have implemented a vectorized Newton-Raphson optimization scheme.
By specifiying a function for computing the log likelihood and parameter initializations
we can use `susiepy.newton_raphson_generator` to get a suite of functions for optimizing
the coefficients of multiple regression problems in a vectorized manner.
Optimization is performed via Newton-Raphson. The gradients and hessians are handled by `jax.grad`.
At each iteration we check for improvement in the log likelihood 
and we halve the step size if the proposed update decreases the log likelihood.
For strictly convex problems, this will be guarunteed to converge to the global optimum 
since the newton step is an increasing direction. 

Optimization is vectorized via `jax.vmap` so we can solve several regression problems simultanesouly. 
Functiosn are JIT compiled using `jax.jit`, which offer substantial speedup after compilation.

### Chunking

Even with the JIT compiled program, run-time can be quite slow for large problems (these reflect hardware limitations e.g. RAM and CPU). If we need to solve `p` regression problems it may be useful to break them up into chunks e.g of size `p/10`. We gain the advantage of the efficient vectorized computation but do not run into the hardware bottle necks. So the output of `newton_raphson_generator` includes a function `fit_vmap_jit_chunked` which allows chunking over the vectorized dimension (columns of `X`). Likewise, `gibbs_generator` returns two function `fit_ser` and `generalized_ibss` which both take an optional argument `n_chunks`.

To illustrate the point, fitting a logistic SER with `n=50000` and `p=100` takes ~3.5 seconds with `n_chunks = 1` and ~0.6 seconds with `n_chunks = 10` on my laptop (8Gb LPDDR3 RAM and 1.4GHz Quad-Core Intel i5 processor).