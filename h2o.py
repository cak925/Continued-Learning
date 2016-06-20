import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h2o

import h2o
import time
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch

# start h2o on local machine
h2o.init()

# load data
def load_data(file):
	return h2o.import_file(path = file)

# check for NA
def is_na(data):
	return data.isna().sum()

def target_hist(data, target):
	data[target].hist(breaks='Sturges')

# train, test split
def train_test_split(data, row_split, target):

	train = data[:row_split, :]
	test = data[row_split:, :]
	return train, test



def gb_grid(train, test):
	ntrees = [10,25,50,75,100]
	max_depth = [2,4,6,10]
	learnrate = [.05, .2,.5,.7]
	min_rows=[5]
	gb_hyper_parameters = {"ntrees":ntrees, "max_depth":max_depth, "learn_rate":learnrate, "min_rows":min_rows}
	gb_model_grid = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params=gb_hyper_parameters)
	gb_model_grid.train(x = train.columns[1:],
					    y = train.columns[0],
					    training_frame = train,
					    validation_frame = test,
					    nfolds=10)

	print gb_model_grid


def dl_grid(train, test):
	activation = ["Tanh","Rectifier","RectifierWithDropout"]
	hidden = [[100,100],[40,20,10], [200,200,200]]
	l1 = [1e-5,1e-4,1e-2]
	dl_hyper_parameters = {"activation":activation, "hidden":hidden,"l1":l1}

	dl_model_grid = H2OGridSearch(H2ODeepLearningEstimator, hyper_parameters=dl_hyper_parameters)
	dl_model_grid.train(x = train.columns[1:],
					    y = train.columns[0],
					    training_frame = train,
					    validation_frame = test,
					    nfolds=10)


	print dl_model_grid




if name == "__main__":
	#data = h2o.load_data("YearPredictionMSD.txt")

	data = load_data("https://s3.amazonaws.com/consultprob/YearPredictionMSD.txt")
	data.head()

	data.shape
	# (515345, 91)

	is_na(data)
	#None

	train = train_test_split(data, 463715, 'C1')[0]
	test = train_test_split(data, 463715, 'C1')[1]

	model_grids(train, test)



	
