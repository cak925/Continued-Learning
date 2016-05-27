import pandas as pd
import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
import sklearn.cluster
from sklearn.cluster import KMeans
from prettytable import PrettyTable

''' 
Load in the data and create dummy variables for the categorical
variables
'''    

def load_data(data):
    d = pd.read_csv(data)
    return d

def dummies(data, column):
    data_new = pd.concat([pd.get_dummies(data[column]), data], axis=1)
    data_dummies = data_new.drop(column, axis=1)
    return data_dummies

'''
Since we are trying to determine who is the safer driver, let's focus
on 'speed' and whether or not they exceeded the acceleration threshold.
I would like to obtain visualizations of these features
'''
# Visualization of Raw Data
    
def dist_plot(data, column, title):
    plt.xlabel(column)
    plt.ylabel('Probability')
    plt.xlim(0,100)
    plt.title(title)
    sns.distplot(data[column])
    plt.show()
    plt.close()

    
def acc_plot(driver_1, driver_2, name_1, name_2):    
    index=1
    bar_width = 3
    opacity = 0.6
    error_config = {'ecolor': '0.3'}
    men_y = plt.bar(index, sum(driver_1['Y'])/driver_1.shape[0], bar_width,
                        alpha=opacity,
                        color='b',
                        error_kw=error_config,
                        label='Yes')
    men_n = plt.bar(index+bar_width, sum(driver_1['N'])/driver_1.shape[0], bar_width,
                        alpha=opacity,
                        color='r',
                        error_kw=error_config,
                        label='No')   
    wom_y = plt.bar(index+3*bar_width, sum(driver_2['Y'])/driver_2.shape[0], bar_width,
                        alpha=opacity,
                        color='b',
                        error_kw=error_config)
    wom_n = plt.bar(index+4*bar_width, sum(driver_2['N'])/driver_2.shape[0], bar_width,
                        alpha=opacity,
                        color='r',
                        error_kw=error_config)  
    plt.legend(loc='best')
    plt.xticks([(i+1)*4 for i in [0,2.3]],[name_1, name_2], size=15)
    plt.ylabel('Percentage of Occurances', fontsize = 20)
    plt.title("Was the acceleration threshold exceeded?", size=20)
    plt.show()
    plt.close()

def line_plot(driver_1, driver_2, column):
    plt.figure()
    d1 = driver_1.reset_index(drop=True)
    d2 = driver_2.reset_index(drop=True)
    d1[column].plot()
    d2[column].plot(style = 'g')
    plt.ylabel(column)
    plt.xlabel('Time')
    plt.title('{s} Comparison'.format(s=column))
    plt.show()


'''
Since variance and sample size are unequal, I'm going to use Welch's ttest 
to compare speed between the two drivers. 

'''

def t_test(driver_1, driver_2, name_1, name_2, column):
    fig = plt.figure()
    driver1_mean = driver_1[column].mean()
    driver2_mean = driver_2[column].mean()

    print '%s Mean CTR: %s' % (name_1, driver_1)
    print '%s Mean CTR: %s' % (name_2, driver_2)
    print 'diff in mean:' , abs(driver1_mean - driver2_mean)
    p_val = scs.ttest_ind(driver_1[column], driver_2[column], equal_var=False)[1]
    print 'p value is:', p_val

'''
P Value is large, thus there is not enough evidence to suggest a difference in speeds between the two drivers.
'''


'''
Let's also compare drivers in terms of how often they exceeded the speed threshold. I will do 
a two sample z test for proportions. 
'''

def z_test(driver_1, n1, driver_2, n2):
    count = np.array([driver_1, driver_2])
    nobs = np.array([n1, n2])
    z, p = proportions_ztest(count, nobs, value=0, alternative = 'larger')
    print ('z-stat = {z} \n p-value = {p}'.format(z=z,p=p))
    

'''
Ha: Alan exceeds the speed threshold more often

z-stat = 8.08808553748 
p-value = 3.03049182068e-16
 
Conclusion: P value small. Reject the null, Evidence suggests he does!

'''

'''

Refering to question 2, let's look at the difference in odometer reading for each trip.

'''

def odom_table(driver_1, driver_2, name_1, name_2):
    t = PrettyTable(['Driver', 'Odometer Difference'])
    t.add_row([name_1, list(driver_1['odometer'])[-1] - driver_1['odometer'][0]])
    t.add_row([name_2, list(driver_2['odometer'])[len(driver_1['odometer'])] - driver_2['odometer'][0]])
    print t

'''
Now I would like to implement a clustering algorithm to determine which trip was driven on the highway.
Speed was the only variable used. I trained my algorithm on the whole data set, then predicted based on 
the average speed for each trip. 
'''

def kmeans_cluster(data, name_1, name_2, group_1, group_2, pred_row1, pred_row2, cent): 
    k_means= cluster.KMeans(n_clusters=2, max_iter=1000, tol=1e-10, init=cent) 
    fit = k_means.fit(data)
    pred_1 = fit.predict(group_1)
    pred_2 = fit.predict(group_2) 
 
    t = PrettyTable(['Driver', 'Trip Label', 'Speed Cluster Center']) 
    t.add_row([name_1, pred_1[pred_row1], k_means.cluster_centers_[0]]) 
    t.add_row([name_2, pred_2[pred_row2], k_means.cluster_centers_[1]]) 
    print t 



if name == "__main__":
    data = load_data('data.csv')
    data.head()
    data.shape
    data.info()
    
    driver_dum = dummies(data, 'driver')
    event_new = dummies(driver_dum, 'event')
    alan = event_new[event_new['Alan'] == 1].drop(['Barbara','Alan'], axis=1).reset_index(drop=True)
    barb = event_new[event_new['Barbara'] == 1].drop(['Barbara','Alan'], axis=1).reset_index(drop=True)
    
    alan.describe()
    barb.describe()
    alan.shape
    barb.shape

    sns.corrplot(alan)
    sns.corrplot(barb)

    dist_plot(alan, 'speed', 'Probability Distribution: Alan\'s Speed')
    dist_plot(barb, 'speed', 'Probability Distribution: Barbara\'s Speed')

    acc_plot(alan, barb, 'Alan', 'Barbara')
    t_test(alan, barb, 'Alan', 'Barb','speed')

    z_test(sum(alan['Y']), len(alan['Y']) , sum(barb['Y']), len(barb['Y']))

    trip_a = alan[alan['trip_id'] == 7].reset_index(drop=True)
    trip_b = barb[barb['trip_id'] == 13].reset_index(drop=True)

    dist_plot(trip_a, 'speed', "Probability Distribution of Alan's Speed During Trip 7")
    dist_plot(trip_b, 'speed', "Probability Distribution of Barbara's Speed During Trip 13")

    line_plot(trip_a,trip_b,'speed')

    odom_table(trip_a, trip_b, 'Alan', 'Barbara')

    sns.corrplot(alan)
    sns.corrplot(barb)

    a_1 = alan[['speed','trip_id']]
    b_1 = barb[['speed','trip_id']]

    a_g = a_1.groupby('trip_id').mean()
    b_g = b_1.groupby('trip_id').mean()
    a_g['trip_id'] = a_g.index
    b_g['trip_id'] = b_g.index

    a_g = a_g.reset_index(drop=True)
    b_g= b_g.reset_index(drop=True)

    a_g = a_g.drop(['trip_id'], axis=1)
    b_g = b_g.drop(['trip_id'], axis=1)

    s = data[['speed']]

    centroids = np.array([[45],[15]], np.float64(2))

    kmeans_cluster(s,'a','b',a_g,b_g,4,2,centroids)

