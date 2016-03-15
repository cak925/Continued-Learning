
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

# Rainfall data for Nashville, measured since 1871
rain = pd.read_csv('rainfall.csv')
rain.head()

# distribution selection

plt.hist(rain.Jan, normed=True, bins=20)
plt.xlabel('Rainfall')
plt.ylabel('Probability')
plt.title('Histogram of Jan. Rainfall')

#looks like possibly a gamma or normal dist. can be potentionally used to fit the data. Leaning Gamma because
# of the right skew. 
# Method of Moments Estimation

xbar = np.mean(rain.Jan)
s = np.std(rain.Jan, ddof = 1)
samp_var = s**2

#approx. alpha and beta for gamma
alpha = xbar**2/samp_var
beta = xbar/samp_var

# Using the estimated parameters, I will plot the distributions on top of the data

gamma_rvs = scs.gamma(a=alpha, scale=1/beta)
norm_rvs = scs.norm(xbar,samp_var)

x_vals = np.linspace(rain.Jan.values.min(), rain.Jan.values.max())
gamma_p = gamma_rvs.pdf(x_vals)
norm_p = norm_rvs.pdf(x_vals)

# Plot them on top of the data
f, ax = plt.subplots()
ax.hist(rain.Jan, bins=20, normed=1)
ax.set_xlabel('Rainfall')
ax.set_ylabel('Probability Density')
ax.set_title('January Rainfall')

ax.plot(x_vals, gamma_p, color='r', label='Gamma',alpha=.6)
ax.plot(x_vals, norm_p, color='g', label='Gamma',alpha=.6)
ax.legend()

# Gamma is the obvious choice here. Normal doesn't match very well at all. Let's double 
# check this by looking at the kstest from scipy.stats. 
    
gamma_fit = scs.kstest(rain.Jan, 'gamma', args = (alpha, beta))

#P value of .01 is significant at the 1% level. not sure if I am using this correctly


#Now I'll write a function to loop through all the months and create similar plots for each month.

def plot_mom(data, col, ax):
    dat = data[col]
    
    xbar = np.mean(dat)
    s = np.std(dat, ddof = 1)
    samp_var = s**2

    alpha = xbar**2/samp_var
    beta = xbar/samp_var

    gamma_rvs = scs.gamma(a=alpha, scale=1/beta)
    x_vals = np.linspace(dat.min(), dat.max())
    gamma_p = gamma_rvs.pdf(x_vals)
    
    ax.plot(x_vals, gamma_p, color='r',alpha=.4)
    ax.set_xlabel('Rainfall')
    ax.set_ylabel('Prob. Density')
    ax.set_title(col)
    
    ax.legend()
    
    label = 'alpha = %.2f\nbeta = %.2f' % (alpha, beta)
    ax.annotate(label, xy=(8, 0.3))
    
months = rain.columns-['Year']
months_df = rain[months]

# Use pandas to get the histogram, the axes as tuples are returned
axes = months_df.hist(bins=20, normed=1,
                    grid=0, edgecolor='none',
                    figsize=(15, 10),
                    layout=(3,4))

# Iterate through the axes and plot the line on each of the histogram

for month, ax in zip(months, axes.flatten()):
    plot_mom(months_df, month, ax)
    
plt.tight_layout()


# likelihood() function that takes in a lambda and a discrete value, 
# and returns the likelihood of observing that value given the particular lambda.

def likelihood(lamb, value):
    return scs.poisson(lamb).pmf(value)
    
# non-parametric estimation, (i.e. making no assumptions about 
# the form of the underlying distribution) using kernel density estimation

dist1 = scs.norm(0,2)
dist2 = scs.norm(4,1)

# sample from each distribution, store in a single array. ie 'merge' dist.
samp1 = dist1.rvs(size=500)
samp2 = dist2.rvs(size=500)

merg_samp = np.append(samp1,samp2)
len(merg_samp)

#plot hist and fit distribution function to our data
x = np.linspace(merg_samp.min(), merg_samp.max())
merg_kde = scs.gaussian_kde(merg_samp)
plt.hist(merg_samp, bins=30, alpha=.5, normed=1)
plt.plot(x,merg_kde(x),'r', label="Gaussian KDE")
plt.legend()
plt.xlabel('Value')
plt.ylabel('Prob Density')
plt.title('KDE of Bimodal Normal Dist')

'''
peaks are right around where I expected them to be (around 0 and 4),
considering we are sampling from a normal(0,2) and a normal(4,1).
A non-parametric vs. parametric approach depends on the bias variance tradeoff you would 
like to allow. Non-Parametric allows a tighter fit to the data, thus higher variance 
but lower bias. Parametric approaches impose bias on your fit because the data might not 
follow the parametric distribution you chose...but variance will generally be lower. 
My method of choice will be dictated by the desired levels of variance and bias, as well
as what I would like to do with my model.
'''
