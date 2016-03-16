import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

#Implement a function which will make a number of draws from a defined distribution.

def draw(distribution, parameters, size):
    if distribution == 'binomial':
        n, p = parameters['n'], parameters['p']
        dist = scs.binom(n,p).rvs(size)
    
    if distribution == 'poisson':
        lamb = parameters['lambda']
        dist = scs.poisson(lamb).rvs(size)
        
    if distribution == 'exponential':
        lamb = parameters['lambda']
        dist = scs.expon(lamb).rvs(size)

    if distribution == 'gamma':
        alpha, beta = parameters['alpha'], parameters['beta']
        dist = scs.gamma(alpha,beta).rvs(size)
        
    if distribution == 'normal':
        mu, var = parameters['mu'], parameters['var']
        dist = scs.norm(mu,var).rvs(size)
        
    return dist
    
# Implement a plot function that bootstraps from the distribution, computes the mean of each
# sample, then makes a histogram of the sample means

def plot_means(dist, parameters, size=200, repeat=5000):
    '''
    - distribution(STR) [specify which distribution to draw from]
    - parameters(DICT) [dictionary with different params depending]
    - size(INT) [the number of values we draw]
    - repeat(INT) [the times we draw values]
    '''
    
    plt.hist([np.mean(draw(dist,parameters, size)) for _ in xrange(repeat)],bins=50)
    
plot_means('normal', {'mu': 10, 'var': 15})
plot_means('poisson', {'lambda': 2})
plot_means('binomial', {'n': 100, 'p': 0.1})
plot_means('exponential', {'lambda': 2})
plot_means('gamma', {'alpha': 0.1, 'beta': 1})
