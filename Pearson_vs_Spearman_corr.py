import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

# Pearson vs. Spearman Correlation

df = pd.read_csv('admissions_with_study_hrs_and_sports.csv')
df.head()

plt.scatter(df.gpa, df.hrs_studied, alpha=.1, edgecolor='blue')
slope, intercept, r_value, p_value, std_err = sc.linregress(df.gpa, df.hrs_studied)
plt.plot(df.gpa, slope*df.gpa + intercept, color='r', alpha=.5)
plt.xlabel('GPA')
plt.ylabel('Hours Studied')

sc.pearsonr(df.gpa, df.hrs_studied)
sc.spearmanr(df.gpa, df.hrs_studied)

plt.scatter(df.gpa, df.sport_performance, alpha=.1, edgecolor='blue')
slope, intercept, r_value, p_value, std_err = sc.linregress(df.gpa, df.sport_performance)
plt.plot(df.gpa, slope*df.gpa + intercept, color='r', alpha=.5)
plt.xlabel('GPA')
plt.ylabel('Sports Performance')

sc.pearsonr(df.gpa, df.sport_performance)
sc.spearmanr(df.gpa, df.sport_performance)

#looking at the graph, we can see there is a strong relationship between gpa and sports performance, but because 
# our coefficients only look at linear and monotonic relationships, the coefficients are low

