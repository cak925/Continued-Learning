import pandas as pd
import numpy as np 
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


# Analyze factors correlated with the GPA of students. A University would like to this information 
# for admitance purposes.

df = pd.read_csv('admissions.csv')
df.head()

# columns are family_income, gpa and parent_avg_age
# I will now implement a covariance function and compute covariance matrix of the df

def covariance(x1,x2):
    return np.sum((x1-np.mean(x1))*(x2-np.mean(x2)))/(len(x1)-1)

print covariance(df.family_income, df.gpa)            
print covariance(df.family_income, df.parent_avg_age)
print covariance(df.gpa, df.parent_avg_age)

# check results
df.cov()

# Next, Compute the correlation matrix from the covariance matrix. 

def correlation(x1,x2):
    return covariance(x1,x2)/(np.sqrt(covariance(x1,x1))*np.sqrt(covariance(x2,x2)))
    
print correlation(df.family_income, df.gpa)
print correlation(df.family_income, df.parent_avg_age)
print correlation(df.gpa, df.parent_avg_age)

# check
df.corr()

# GPA and income are strongly and positively correlated.

# The university would like to make sure people of all family income ranges are being represented.
# Thus three GPA thresholds will be set according to income. Groups are 
# 0-26,832 ... 26,833-37,510 ... 37,511 - 51,1112
# I will now write a function that would plot the distribution of GPA scores for each family income category.
# ie. the conditional probability distributions of gpa given certain levels of family income

#First, I will categorize the income

def income_cat(income):
    if income <= 26832:
        return 'low'
    elif income <= 37510:
        return 'medium'
    else:
        return 'high'

#Apply categorication and make a new column in the data frame

df['family_income_cat'] = df.family_income.apply(income_cat)

df.head()

# Get the conditional distribution of GPA given an income class

low_income_gpa = df[df['family_income_cat'] == 'low'].gpa
med_income_gpa = df[df['family_income_cat'] == 'medium'].gpa
high_income_gpa = df[df['family_income_cat'] == 'high'].gpa

#plot distributions

def plot_dist(gpa_sample, label):
    my_pdf = gaussian_kde(gpa_sample)
    x = np.linspace(min(gpa_sample), max(gpa_sample))
    plt.plot(x, my_pdf(x), label=label)
    
fig = plt.figure(figsize=(12,5))
plot_dist(low_income_gpa, 'low income')
plot_dist(med_income_gpa, 'medium income')
plot_dist(high_income_gpa, 'high income')
plt.xlabel('GPA')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# The university would like to admit the 90th percentile for each income class. What is that GPA?

print '90th percentile GPA for low income class', np.percentile(low_income_gpa, 90)
print '90th percentile GPA for medium income class', np.percentile(med_income_gpa, 90)
print '90th percentile GPA for high income class', np.percentile(high_income_gpa, 90)

