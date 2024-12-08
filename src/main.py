import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split


def load_data(fname):
    df = pd.read_csv(fname)
    nc = df.shape[1] # number of columns
    matrix = df.values # Convert dataframe to darray
    table_X = matrix [:, 0:nc-1] # get features 
    table_y = matrix [:, nc-1] # get class (last columns)           
    features = df.columns.values[0:nc-1] # get features names
    target = df.columns.values[nc-1] # get target name
    return table_X, table_y, features, target


def remove_feature(table_X, features, name):
    i = np.where(features == name)[0][0]
    return np.delete(table_X, i, 1), np.delete(features, i, 0)


def remove_features(table_X, features, names):
    for n in names:
        table_X, features = remove_feature(table_X, features, n)
    return table_X, features


def print_matrix(array):
    for i in array:
        print(i)


table_X, table_y, features, target = load_data("../data/PetFinder_dataset.csv")


features_to_remove = ['Name', 'PetID', 'Description', 'RescuerID']
table_X, features = remove_features(table_X, features, features_to_remove)

print_matrix(table_X)

table_X = table_X.astype(float)
table_y = table_y.astype(float)

train_X, test_X, train_y, test_y = train_test_split(table_X, table_y, random_state=0)

dtc_Gini = tree.DecisionTreeClassifier(random_state = 0, criterion="gini", splitter='best')
dtc_Gini = dtc_Gini.fit(train_X, train_y)
print(dtc_Gini.score(test_X, test_y))

