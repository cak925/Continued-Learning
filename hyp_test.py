import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import math
from itertools import combinations
'''
Use hypothesis testing to analyze Click Through Rate on the NYT website.
Is there a statistically significant difference between mean CTR for signed in
users and users who are not signed in, male vs. female, each of 7 age groups against 
eachother
'''
ls

data = pd.read_csv('data/nyt1.csv')
data.head()
data.info()
data.shape
# adjustment needed for to account for multiple testing at the .05 significance level

comb = len(list(combinations(range(1,8),2)))

# 23 tests total

alpha = .05/23

data = data[data.Impressions != 0]
data['CTR'] = data['Clicks']/data['Impressions'].astype(float)
data.columns

#function to plot histogram of columns

def ctr_hist(df, title):
    df.hist(figsize=(12,5), grid=False, normed=True, alpha=.2)
    plt.suptitle(title, size=18, weight='bold', y=1.05)

ctr_hist(data, 'Overall')

# separate data frame. users who are signed in, users who are not.

sig_in = data[data['Signed_In'] == 1]
not_sig_in = data[data['Signed_In'] == 0]

ctr_hist(sig_in, "signed in users")
ctr_hist(not_sig_in, "users not signed in")

#CTR between signed in and non signed in users does not visually
# look different. I am going to perform a t test to see if they 
# are statistically different
s = sig_in['CTR'].mean()
n = not_sig_in['CTR'].mean()
diff = s-n
diff
scs.ttest_ind(sig_in['CTR'], not_sig_in['CTR'], equal_var=False)
# pvalue of 0 indicates they are significantly different

# what about the difference between males and females, for the signed in users?

male = sig_in[sig_in['Gender'] == 1]
female = sig_in[sig_in['Gender'] == 0]

male['CTR'].mean()
female['CTR'].mean()

scs.ttest_ind(male['CTR'], female['CTR'], equal_var=False)
male['CTR'].mean()
# pvalue .001, alpha is .002. thus still significantly different
# but this may require further investigation since this difference
# is only marginally significant
