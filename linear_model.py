import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_predict,cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from auxiliary import every_nth

file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)            # data as pandas DataFrame format
attribute_names = df_data.columns.values    # numpy array of attribute names
df_data.sample(frac=1)                      # shuffle rows of data
data = df_data.values                       # all data in numpy array format

vf = 0.2                                    # fraction of the data reserved for validation (usually between 0.2 and 0.3)
N = data.shape[0]                           # number of rows in the dataset
M = data.shape[1]                           # number of columns in the dataset
y = data[:, 10]                             # class belonging to each row in normal format
X = np.delete(data, 10, axis=1)

# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)

# create model
lin_model = lm.LinearRegression()


# the test set we keep for testing
# the training set we use k-folds cross validation
folds = 15
scores = cross_val_score(lin_model, X_train, y_train, cv=folds)
print(scores)

predictions = cross_val_predict(lin_model, X_test, y_test, cv=folds)
plt.scatter(y_test, predictions, marker='x')
plt.xlabel("True Tempo")
plt.ylabel("Predicted Tempo")
plt.show()

accuracy = metrics.r2_score(y_test, predictions)
print("Accuracy: {}".format(accuracy))
