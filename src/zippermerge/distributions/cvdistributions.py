"""
cvdistributions.py

Cryptographically secure random number generation module.

Functions:
uniform(a, b)
    Cryptographically secure uniform sample in [a, b).

exponentialdist(lam)
    Generate a single cryptographically secure random sample from an
    Exponential(lam) distribution using Inverse Transform Sampling.

poissondist(lam)
    Generate a single cryptographically secure random sample from a
    Poisson(lam) distribution using Inverse Transform Sampling for a
    discrete distribution.

"""

import secrets
import math


def uniform(a: float = 0.0, b: float = 1.0) -> float:
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53)  
    return a + (b - a) * u


def exponentialdist(lam: float) -> float:
    """
    CDF of Exponential distribution: 
        F(x) = 1 - exp(-lam * x), x >= 0
    """
    y = uniform(0.0, 1.0)

    while y == 0.0:
        y = uniform(0.0, 1.0)

    x = -(1.0 / lam) * math.log(y)
    return x


def poissiondist(lam: float) -> int:
    """
    Poisson CDF:
        F(k) = P(X <= k) = sum_{i=0}^{k} exp(-lam) * lam^i / i!,  k = 0, 1, 2, ...
    """
    u = uniform(0.0, 1.0)

    p = math.exp(-lam)   
    cumulative = p       
    k = 0

    while u > cumulative:
        k += 1
        p *= lam / k       
        cumulative += p

    return k
