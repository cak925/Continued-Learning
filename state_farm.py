import pandas as pd
import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
import sklearn.cluster
from sklearn.cluster import KMeans
from prettytable import PrettyTable
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(data, response):
    d = pd.read_csv(data)
    y = d[response]
    X = d.drop(response, axis=1)
    return X, y

X, y = load_data('data_sf.csv', 'X1')
X.head()
X.shape
X.info()

def miss_values(data):
	return data.isnull().sum()


def dummies(data, column):
    data_new = pd.concat([pd.get_dummies(data[column]), data], axis=1, inplace=True, dummy_na=True)
    data_dummies = data_new.drop(column, axis=1)
    return data_dummies

# x7, x8, x9, x12, x17, x32

def strip_col(col, strip):
    new = [float(str(l).translate(None, strip)) for l in col] 
    return new
    
X['X4'] = strip_col(X['X4'],'$,')
X['X5'] = strip_col(X['X5'],'$,')
X['X6'] = strip_col(X['X6'],'$,')
X['X30'] = strip_col(X['X30'],'%')
X['X19'] = strip_col(X['X19'], 'xx')
X['X11'] = strip_col(X['X11'], '< years + n/')
y = strip_col(y,'%')

X.isnull().sum()
X[X['X4'].isnull()] # Row 364111 is completely Null for all variables, drop it!

X = X.drop(364111).reset_index()
X.isnull().sum()

def check_unique(data,col):
	return data[col].unique()

check_unique(X,'X7')	# 2
check_unique(X,'X8')	# 8 incl nan
check_unique(X,'X9')	# 36 incl nan
check_unique(X,'X12')	# 7, ['RENT', 'OWN', 'MORTGAGE', 'NONE', nan, 'OTHER', 'ANY']
check_unique(X,'X17')	# 14
check_unique(X,'X32')	# 2

X = dummies(X,'X7') #done
X = dummies(X,'X8') # done
X = dummies(X, 'X9') # done
X = dummies(X, 'X12') # done
X = dummies(X, 'X17') # done
X = dummies(X,'X32') # done 

x1 = [pd.to_datetime(j, unit="D", format='%b-%Y') for j in X['X23']]

dt_1 = [(datetime.datetime.strptime(j, "%b-%y") for j in X['X23'])]
dt_2 = [(datetime.datetime.strptime(k, "%d-%b") for j in X['X23'])]
if dt > datetime.now():
    dt = dt - datetime.timedelta(years=100)

def to_date(data, col):
	dates = []
	for i in xrange(len(data[col])):
		if data[col][i][0].isdigit() == True:
			dates.append(datetime.strptime(data[col][i], "%d-%b"))
		else: 
			dates.append(datetime.strptime(data[col][i], "%b-%y"))

	data[col] = dates
	return data 

for i in X['X11']:
	

