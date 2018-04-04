import numpy as np

def stochastic_gradient_descent(grad, data, d, w = None, eta = 0.01, iters = 100):
    if w is None: w = np.zeros(d)
    for i in range(iters):
        for p in data:
            w = w - eta * grad(p, w)
    return w

def G(xy, w):
    x, y = xy
    return 2*(w.dot(x) - y)*x
