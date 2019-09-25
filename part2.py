# IMPORT MODULES:
import pandas as pd
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

# IMPORT CLASSIFICATION ALGORITHMS:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# STOPS 'FUTURE WARNING' FOR VERSION CHANGES
import warnings
warnings.filterwarnings('ignore')

# INIT SETTINGS:
seed = 309
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3

# TRAINING SETTINGS:
alpha = 0.1  # step size
max_iterations = 100  # max iterations

# ENSURES ALL DATA COLUMNS ARE PRINTED TO CONSOLE
pd.set_option('display.width', 300)
np.set_printoptions(linewidth=300)
pd.set_option('display.max_columns', 10)

# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


def load_data():
    columns = ['age', 'workClass', 'finalWeight', 'educationName', 'Education', 'maritalStatus',
               'occupation', 'relationship', 'race', 'sex', 'capGain', 'capLoss', 'hoursPerWeek', 'country',
               'salary']

    df_train = pd.read_csv("adult.data", header=None, names=columns, delimiter=' *, *')
    df_test = pd.read_csv("adult.test", skiprows=[0], header=None, names=columns, delimiter=' *, *')

    df_train = df_train.replace('?', np.nan)
    df_train.dropna(how='any', inplace=True)
    df_test = df_test.replace('?', np.nan)
    df_test.dropna(how='any', inplace=True)

    return df_train, df_test


def process_data(train, test):

    # CHECK FOR MISSING VALUES
    if train.isnull().values.any() or test.isnull().values.any():
        train.dropna(inplace=True)  # REMOVES MISSING VALUES IF ANY PRESENT.
        test.dropna(inplace=True)  # REMOVES MISSING VALUES IF ANY PRESENT.
        print('\nMISSING VALUES HAVE BEEN REMOVED:\n')
    else:
        print('\nMISSING VALUES:', train.isnull().values.any(), '\n')
        print('\nMISSING VALUES:', test.isnull().values.any(), '\n')

    train = train.replace(">50K", 1)
    train = train.replace("<=50K", 0)
    test = test.replace(">50K.", 1)
    test = test.replace("<=50K.", 0)

    train = train.drop('educationName', 1)
    test = test.drop('educationName', 1)

    replace_categorical = {"workClass":     {'Private': 1, 'Self-emp-not-inc': 2, 'Self-emp-inc': 3, 'Federal-gov': 4,
                                             'Local-gov': 5, 'State-gov': 6, 'Without-pay': 7, 'Never-worked': 8},
                           "maritalStatus": {'Married-civ-spouse': 1, 'Divorced': 2, 'Never-married': 3, 'Separated': 4,
                                             'Widowed': 5, 'Married-spouse-absent': 6, 'Married-AF-spouse': 7,
                                             'Never-worked': 8},
                           "occupation":    {'Tech-support': 1, 'Craft-repair': 2, 'Other-service': 3, 'Sales': 4,
                                             'Exec-managerial': 5, 'Prof-specialty': 6, 'Handlers-cleaners': 7,
                                             'Machine-op-inspct': 8, 'Adm-clerical': 9, 'Farming-fishing': 10,
                                             'Transport-moving': 11, 'Priv-house-serv': 12, 'Protective-serv': 13,
                                             'Armed-Forces': 14},
                           "relationship":  {'Wife': 1, 'Own-child': 2, 'Husband': 3, 'Not-in-family': 4,
                                             'Other-relative': 5, 'Unmarried': 6},
                           "race":          {'White': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4,
                                             'Black': 5},
                           "sex":           {'Female': 0, 'Male': 1},
                           "country":       {'United-States': 1, 'Cambodia': 2, 'England': 3, 'Puerto-Rico': 4,
                                             'Canada': 5, 'Germany': 6, 'Outlying-US(Guam-USVI-etc)': 7, 'India': 8,
                                             'Japan': 9, 'Greece': 10, 'South': 11, 'China': 12, 'Cuba': 13, 'Iran': 14,
                                             'Honduras': 15, 'Philippines': 16, 'Italy': 17, 'Poland': 18,
                                             'Jamaica': 19, 'Vietnam': 20, 'Mexico': 21, 'Portugal': 22, 'Ireland': 23,
                                             'France': 24, 'Dominican-Republic': 25, 'Laos': 26, 'Ecuador': 27,
                                             'Taiwan': 28, 'Haiti': 29, 'Columbia': 30, 'Hungary': 31, 'Guatemala': 32,
                                             'Nicaragua': 33, 'Scotland': 34, 'Thailand': 35, 'Yugoslavia': 36,
                                             'El-Salvador': 37, 'Trinadad&Tobago': 38, 'Peru': 39, 'Hong': 40,
                                             'Holand-Netherlands': 41}}

    train.replace(replace_categorical, inplace=True)
    test.replace(replace_categorical, inplace=True)

    return train, test


def compute(train_xx, test_xx, train_yy, test_yy, function, name):

    start_time = datetime.datetime.now()

    function.fit(train_xx, train_yy)

    prediction = function.predict(test_xx)

    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    accuracy = accuracy_score(test_yy, prediction)
    precision = precision_score(test_yy, prediction)
    recall = recall_score(test_yy, prediction)
    f1 = f1_score(test_yy, prediction)
    auc = roc_auc_score(test_yy, prediction)

    print("\n||=======================================||\n")
    print("Classifier Algorithm: " + name)
    print("Time             : {t:.2f} seconds".format(t=execution_time))
    print('Accuracy         : %0.2f ' % accuracy)
    print('Precision        : %0.2f ' % precision)
    print('Recall           : %0.2f ' % recall)
    print('F1-Score         : %0.2f ' % f1)
    print('Area Under Curve : %0.2f ' % auc)


if __name__ == '__main__':

    # IMPORT DATA SET
    train_data, test_data = load_data()

    print('\nSHAPE:', test_data.shape)
    print('\nSHAPE:', train_data.shape)

    # VIEW INITIAL DATA HEAD
    print('\nHEAD | No pre-processing\n', train_data.head(5))

    train_data, test_data = process_data(train_data, test_data)

    train_x = train_data.drop(['salary'], axis=1)
    test_x = test_data.drop(['salary'], axis=1)
    train_y = train_data['salary']
    test_y = test_data['salary']

    compute(train_x, test_x, train_y, test_y, KNeighborsClassifier(), 'K Neighbors Classifier')
    compute(train_x, test_x, train_y, test_y, GaussianNB(), 'Gaussian Naive Bayes')
    compute(train_x, test_x, train_y, test_y, SVC(), 'SVM')
    compute(train_x, test_x, train_y, test_y, DecisionTreeClassifier(), 'Decision Tree')
    compute(train_x, test_x, train_y, test_y, RandomForestClassifier(), 'Random Forest')
    compute(train_x, test_x, train_y, test_y, AdaBoostClassifier(), 'ADA Boost')
    compute(train_x, test_x, train_y, test_y, GradientBoostingClassifier(), 'Gradient Boosting')
    compute(train_x, test_x, train_y, test_y, LinearDiscriminantAnalysis(), 'Linear Discriminant Analysis')
    compute(train_x, test_x, train_y, test_y, MLPClassifier(), 'Multi-Layer Perceptron')
    compute(train_x, test_x, train_y, test_y, LogisticRegression(), 'Logistic Regression')


    # categorical columns:

    # workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    # marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    # occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners,
    # Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    # relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    # race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    # sex: Female, Male.
    # native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    # India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    # Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
    # Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
