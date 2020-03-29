import numpy as np
import pandas as pd
import graphviz as gr
import sklearn.tree as tree
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from scipy.stats import skewnorm, norm
from sklearn import metrics
import matplotlib.pyplot as plt
from auxiliary import get_percentiles
from sklearn import preprocessing
from auxiliary import one_out_of_k, calc_distribution, trim_ticks

file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)  # data as pandas DataFrame format
attribute_names = df_data.columns.values  # numpy array of attribute names
df_data.sample(frac=1)  # shuffle rows of data
data = df_data.values  # all data in numpy array format

vf = 0.2  # fraction of the data reserved for validation (usually between 0.2 and 0.3)
N = data.shape[0]  # number of rows in the data set
M = data.shape[1]  # number of columns in the data set
y = data[:, 13]  # class belonging to each row in normal format
X = np.delete(data, 13, axis=1)
X = preprocessing.scale(X)
folds = 10  # fold for k-folds x-validation

# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)

weight_dict = dict(zip([1, 2, 3, 4, 5], [1, 1, 1, 3, 5]))   # pre-defined weights for each class
criterion = 'gini'
min_sample = 20
dec_tree = tree.DecisionTreeClassifier(criterion=criterion,
                                       min_samples_split=min_sample,
                                       class_weight=weight_dict)  # create model
dec_tree.fit(X_train, y_train)
# the test set we keep for testing
# the training set we use k-folds cross validation
scores = cross_val_score(dec_tree, X_train, y_train, cv=folds)  # calculates x-validation scores
print("X-validation used: %i-folds" % folds)
print("X-validation accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

predictions = cross_val_predict(dec_tree, X_test, y_test, cv=folds)  # calculates the test accuracy
classification_report = metrics.classification_report(y_test, predictions)  # calculates the R^2 value between true- & predicted values
print(classification_report)

#tree.export_graphviz(dec_tree, out_file="decition_tree_model.dot")
#with open("decition_tree_model.dot") as f:
#    dot_graph = f.read()
#gr.Source(dot_graph)

#tree.plot_tree(dec_tree)
#plt.show()