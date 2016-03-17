import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import math

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

# Population Inference/Confidence Intervals

# length of lunch break, 200 google employees
lunch = np.loadtxt('lunch_hour.txt')

plt.hist(lunch, normed=True)
plt.xlabel('Hours')
plt.ylabel('Probability Density')

s= np.std(lunch)
xbar = np.mean(lunch)

# Because sample size is > 30, we can assume the sampling dist. of the sample mean is 
# approx normal. Thus we can use the normal dist. formula for a CI (CLT)

conf_int = scs.norm.interval(.95, loc=xbar, scale=s/np.sqrt(len(lunch)))
#(2.1044876359816542, 2.2645123640183455)
# 95% chance an employee will take a lunch beteen 2.1 and 2.26 hours
# DAMN that's a long lunch!

'''
Bootstrapping! Useful when the underlying pop. dist is unknown, or if the sample
size is small. With this small sample size, using bootstrapping we can compute a more 
precises 95% CI because we are not making any distributional assumps. with the bootstrapping
distribution. At small sample sizes, CLT does not apply.
'''

# Implement a bootstrap function to randomly draw, with replacement, from a sample. 

def bootstrap(data, iterations=10000):
    if type(data) != np.ndarray:
        data = np.array(data)
        
    return [[np.random.choice(data,size=len(data), replace=True)] for _ in xrange(iterations)]
    

# CI for mean using boostrapping

# Will changing to Apple monitors increase productivity? 

# difference in productivity, before and after the change
productivity = np.loadtxt('productivity.txt')

#implement a bootstrap ci function to calculate the ci for the sample mean

def bootstrap_ci(data, iterations=1000, ci=95):
    boot_samples = bootstrap(data)
    means = [np.mean(i) for i in boot_samples]
    low_bound = (100-ci)/2
    high_bound = 100 - low_bound
    lower_ci, upper_ci = np.percentile(means, [low_bound, high_bound])
    return lower_ci, upper_ci

bootstrap_ci(productivity)
# (-0.34015999999999985, 10.475999999999999)

m = [np.mean(i) for i in bootstrap(productivity)]
plt.hist(m)
plt.xlabel('Difference in Productivity')
plt.ylabel('Frequency')

