from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import matplotlib.pyplot as plt
import numpy as np


bal = pd.read_csv('data/balance.csv')

X = bal.drop('Balance', axis=1)

X_upper= X[X['Limit'] >= 3000 ].drop('Unnamed: 0', axis=1)
X_upper.head()
X_lower= X[X['Limit'] < 3000 ].drop('Unnamed: 0', axis=1)
y=bal[['Balance','Limit']]
y_upper = np.ravel(y[y['Limit'] >= 3000 ].drop('Limit', axis=1))
y_lower = np.ravel(y[y['Limit'] < 3000 ].drop('Limit', axis=1))

# Married - 1, Not Married - 0
X_upper['Married'] = pd.get_dummies(X_upper['Married'])
# Female - 0, Male - 1
X_upper['Gender'] = pd.get_dummies(X_upper['Gender'])

# Student Yes - 1, Student No - 0
X_upper['Student'] = pd.get_dummies(X_upper['Student'])


# Get the Dummy variables
ethnicity_dummy = pd.get_dummies(X_upper['Ethnicity'])
# Only need two of the three values
X_upper[ ['Asian', 'Caucasian'] ] = ethnicity_dummy[ ['Asian', 'Caucasian'] ]
# Remove the Ethnicity column
del X_upper['Ethnicity']
X_upper.head()

train_x, test_x, train_y, test_y = train_test_split(X_upper, y_upper, test_size=.2, random_state=1)

rf_grid = {'max_depth': [1,2,3,None],
           'max_features': [1, 3, 10],
           'min_samples_split': [1, 3, 5, 10],
           'min_samples_leaf': [1, 3, 10,50],
           'bootstrap': [True, False],
           'n_estimators': [100,500,1000,1500],
           'random_state': [1]}

gd_grid = {'learning_rate': [0.1,.3,.6],
            'min_samples_split':[1,2,3,4],
           'max_depth': [1,2,4],
           'min_samples_leaf': [3, 5],
           'max_features': [7,8,9],
           'n_estimators': [1500,2000],
           'random_state': [1]}

def grid_search(est, grid):
    grid_cv = GridSearchCV(est, grid, n_jobs=-1, verbose=True,
                           scoring='mean_squared_error').fit(train_x, train_y)
    return grid_cv

rf_grid_search = grid_search(RandomForestRegressor(), rf_grid)
gd_grid_search = grid_search(GradientBoostingRegressor(), gd_grid)

rf_best = rf_grid_search.best_estimator_
gd_best = gd_grid_search.best_estimator_




rf_grid_search.best_params_
'''
{'bootstrap': False,
 'max_depth': None,
 'max_features': 10,
 'min_samples_leaf': 1,
 'min_samples_split': 1,
 'n_estimators': 100,
 'random_state': 1}
'''

gd_grid_search.best_params_
"""
{'learning_rate': 0.6,
 'max_depth': 1,
 'max_features': 8,
 'min_samples_leaf': 5,
 'min_samples_split': 1,
 'n_estimators': 1500,
 'random_state': 1}
"""

def cross_val(estimator, train_x, train_y):
    # n_jobs=-1 uses all the cores on your machine
    mse = cross_val_score(estimator, train_x, train_y,
                           scoring='mean_squared_error',
                           cv=10, n_jobs=-1) * -1
    r2 = cross_val_score(estimator, train_x, train_y,
                           scoring='r2', cv=10, n_jobs=-1)

    mean_mse = mse.mean()
    mean_r2 = r2.mean()

    params = estimator.get_params()
    name = estimator.__class__.__name__
    print '%s Train CV | MSE: %.3f | R2: %.3f' % (name, mean_mse, mean_r2)
    return mean_mse, mean_r2 
    
cross_val(gd_best, train_x, np.array(train_y))
cross_val(rf_best, train_x, np.array(train_y))

cross_val(gd_best, test_x, test_y)
cross_val(rf_best, test_x, test_y)


col_names = X_upper.columns
# sort importances
indices = np.argsort(gd_best.feature_importances_)
# plot as bar chart
figure = plt.figure(figsize=(10,7))
plt.barh(np.arange(len(col_names)),gd_best.feature_importances_[indices],
         align='center', alpha=.5)
plt.yticks(np.arange(len(col_names)), np.array(col_names)[indices], fontsize=14)
plt.xticks(fontsize=14)
_ = plt.xlabel('Relative importance', fontsize=18)

fig, axs = plot_partial_dependence(gd_best, train_x, range(X_upper.shape[1]) ,
                                   feature_names=col_names, figsize=(15, 10))
fig.tight_layout()
