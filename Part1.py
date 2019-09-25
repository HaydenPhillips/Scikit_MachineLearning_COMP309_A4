# IMPORT MODULES:
import pandas as pd
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

# IMPORT REGRESSION ALGORITHMS:
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# STOPS 'FUTURE WARNING' FOR VERSION CHANGES
# import warnings
# warnings.filterwarnings('ignore')

# INIT SETUP:
seed = 309
random.seed(seed)
np.random.seed(seed)
train70_test30_split = 0.3  # Slips data: test set = 30%, training set = 70%

# TRAINING SETUP:
alpha = 0.1  # step size
max_iterations = 100

# PRINTS DATA TO CONSOLE
pd.set_option('display.width', 300)
np.set_printoptions(linewidth=300)
pd.set_option('display.max_columns', 20)


def load_data():
    datafile = pd.read_csv("diamonds.csv")
    return datafile


def pre_process_data(data):

    # DROP UNNAMED ATTRIBUTE
    data.drop(['Unnamed: 0'], axis=1, inplace=True)

    # FIND AND REMOVE MISSING VALUES
    data.dropna(inplace=True)   # "Return a new Series with missing values removed"

    # REPLACE CATEGORICAL WITH NUMERICAL DATA
    cat_to_num = {
        "cut":     {"Ideal": 5,  "Premium": 4,   "Very Good": 3, "Good": 2,  "Fair": 1},
        "clarity": {"IF": 8,     "VVS1": 7,      "VVS2": 6,      "VS1": 5,   "VS2": 4,   "SI1": 3,   "SI2": 2, "I1": 1},
        "color":   {"D": 7,      "E": 6,         "F": 5,         "G": 4,     "H": 3,     "I": 2,     "J": 1}
    }
    data.replace(cat_to_num, inplace=True)

    # # PERFORM LOG TRANSFORMATION GIVEN SKEWNESS OF DATA
    # np.log(data['price'])  # Y = FOR AFTER LOG TRANSFORM DIST PLOT
    # data.drop(['price'], axis=1)  # X = FOR AFTER LOG TRANSFORM PLOT

    # SPLIT TRAIN AND TEST DATA
    # train_data, test_data = train_test_split(data, test_size = train70_test30_split)

    # STANDARDISE DATA
    mean = data.mean()
    std = data.std()
    processed_data = (data-mean)/std

    return processed_data


def visualise(data, y):

    # SCATTER PLOT OF ONE FEATURE VS. PRICE
    x_axes = data[y]
    y_axes = data['price']
    # plt.title(y + ' vs. price')
    # plt.scatter(x_axes, y_axes)
    # plt.show()

    # SCATTER PLOT OF ALL FEATURES VS. PRICE
    data.plot.scatter(x='x', y='y', c='r')
    plt.show()

   #  # HISTOGRAM OF FEATURE COUNTS
    data.hist(alpha=0.5, bins=10, figsize=(20, 15))
    plt.show()


    # HISTOGRAM WITH DISTRIBUTION CURVE TO CHECK FOR SKEW
    # The y axis in a density plot is the probability density function for the kernel density estimation.
    sns.distplot(data['price'])
    plt.ylabel('Probability Density')
    plt.xlabel('Price')
    plt.show()
   # #
   #  # AFTER LOG TRANSFORM TAKEN PLACE
    sns.distplot()

    # BOX AND WHISKERS PLOT
    #sns.catplot(data=X, size=5, kind='box')  #,ASPECT=3 !!!


def compute(train_xx, test_xx, train_yy, test_yy, function, name):

    start_time = datetime.datetime.now()

    function.fit(train_xx, train_yy)

    prediction = function.predict(test_xx)

    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    mse = mean_squared_error(test_yy, prediction)
    root_mse = sqrt(mean_squared_error(test_yy, prediction))
    mae = mean_absolute_error(test_yy, prediction)
    r_square = r2_score(test_yy, prediction)

    print("\n||=======================================||\n")
    print("Regression Algorithm: " + name)
    print('MSE          : %0.2f ' % mse)
    print('Root MSE     : %0.2f ' % root_mse)
    print('MAE          : %0.2f ' % mae)
    print('R-Square     : %0.2f ' % r_square)
    print("Time         : {t:.2f} seconds".format(t=execution_time))


if __name__ == '__main__':

    # IMPORT DATA SET
    data = load_data()

    # VIEW INITIAL DATA HEAD
    print('\nHEAD | No pre-processing\n', data.head(3))

    data = pre_process_data(data)

    # VIEW PRE-PROCESSED DATA HEAD
    print('HEAD | With pre-processing\n', data.head(3))

    print('\nSHAPE:', data.shape)
    print('\nDESCRIBE:\n', data.describe())
    print('\nDATA SKEW\n', data.skew(axis=0, skipna=True))

    corr = data.corr()
    print('\nCORRELATION:\n', corr)

    visualise(data, 'carat')  # HIGHEST CORRELATED FEATURE = CARAT

    features_only = data.drop(['price'], axis=1)
    predicted = (data['price'])

    train_x, test_x, train_y, test_y = train_test_split(features_only, predicted,
                                                        random_state=seed, test_size=train70_test30_split)

    compute(train_x, test_x, train_y, test_y, LinearRegression(), 'Linear Regression')
    compute(train_x, test_x, train_y, test_y, KNeighborsRegressor(), 'K Neighbors Regression')
    compute(train_x, test_x, train_y, test_y, Ridge(), 'Ridge Regression')
    compute(train_x, test_x, train_y, test_y, DecisionTreeRegressor(), 'Decision Tree Regression')
    compute(train_x, test_x, train_y, test_y, RandomForestRegressor(), 'Random Forest Regression')
    compute(train_x, test_x, train_y, test_y, RandomForestRegressor(), 'Random Forest Regression')
    compute(train_x, test_x, train_y, test_y, GradientBoostingRegressor(), 'Gradient Boosting Regression')
    compute(train_x, test_x, train_y, test_y, SGDRegressor(), 'SGD Regression')
    compute(train_x, test_x, train_y, test_y, SVR(), 'Support Vector Regression')
    compute(train_x, test_x, train_y, test_y, LinearSVR(), 'Linear Support Vector Regression')
    compute(train_x, test_x, train_y, test_y, MLPRegressor(), 'Multi-Layer Perceptron Regression')