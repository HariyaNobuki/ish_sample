import numpy as np

def addLinker(x, y, z):
    return x+y+z

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

def sin(x1):
    return np.sin(x1)

def cos(x1):
    return np.cos(x1)

def exp(x1):
    return np.expm1(np.abs(x1))

def log(x1):
    return np.log(np.abs(x1))
