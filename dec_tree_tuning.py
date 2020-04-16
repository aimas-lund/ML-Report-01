import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from auxiliary import one_out_of_k, calc_distribution, trim_ticks
from imblearn.under_sampling import RandomUnderSampler

file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)  # data as pandas DataFrame format
attribute_names = df_data.columns.values  # numpy array of attribute names
df_data.sample(frac=1)  # shuffle rows of data
data = df_data.values  # all data in numpy array format

def weight_tuning_counter(weight):
    return list(str(weight))

vf = 0.2  # fraction of the data reserved for validation (usually between 0.2 and 0.3)
#N = data.shape[0]  # number of rows in the data set
#M = data.shape[1]  # number of columns in the data set
y = data[:, 13]  # class belonging to each row in normal format
X = np.delete(data, 13, axis=1)
X = preprocessing.scale(X)
folds = 10  # fold for k-folds x-validation

rus = RandomUnderSampler(random_state=0)
X, y = rus.fit_resample(X, y)
N = X.shape[0]  # number of rows in the data set
M = X.shape[1]  # number of columns in the data set

# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)

#weight_dict = dict(zip([1, 2, 3, 4, 5], [1, 1, 1.05, 1.5, 4]))  # pre-defined weights for each class
criterion = 'gini'
min_sample = 20
dec_tree = tree.DecisionTreeClassifier(criterion=criterion,
                                       min_samples_split=min_sample)  # create model
dec_tree.fit(X_train, y_train)