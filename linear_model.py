import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from scipy.stats import skewnorm, norm
from sklearn import metrics
import matplotlib.pyplot as plt


def get_percentiles(x, lower=10., upper=90.):
    return np.percentile(np.array(x), lower), np.percentile(np.array(x), upper)


def get_limits(x):
    return min(x), max(x)


file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)  # data as pandas DataFrame format
attribute_names = df_data.columns.values  # numpy array of attribute names
df_data.sample(frac=1)  # shuffle rows of data
data = df_data.values  # all data in numpy array format

vf = 0.2  # fraction of the data reserved for validation (usually between 0.2 and 0.3)
N = data.shape[0]  # number of rows in the data set
M = data.shape[1]  # number of columns in the data set
y = data[:, 10]  # class belonging to each row in normal format
X = np.delete(data, 10, axis=1)
folds = 15  # fold for k-folds x-validation

# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)
lin_model = lm.LinearRegression()  # create model

# the test set we keep for testing
# the training set we use k-folds cross validation
scores = cross_val_score(lin_model, X_train, y_train, cv=folds)  # calculates x-validation scores
print("X-validation used: %i-folds" % folds)
print("X-validation accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

predictions = cross_val_predict(lin_model, X_test, y_test, cv=folds)  # calculates the test accuracy
accuracy = metrics.r2_score(y_test, predictions)  # calculates the R^2 value between true- & predicted values
print("Test Accuracy: %0.5f" % accuracy)

# plot true- vs. predicted values
plt.scatter(y_test, predictions, marker='x')
plt.xlabel("True Tempo")
plt.ylabel("Predicted Tempo")
plt.show()

# plot histogram of predicted and true tempos
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
titles = ["True", "Predicted"]
tempos = [y_test, predictions]

norm_param = []
x_dist = []
y_dist = []
x_limits_dist = []

for data in tempos:
    norm_param.append(skewnorm.fit(data))
    x_limits_dist.append(get_percentiles(data, 0.001, 99.99))

for idx, ax in enumerate(axs):
    ax.hist(tempos[idx], bins=30, density=True, facecolor="#0000FF")
    x_dist.append(np.linspace(x_limits_dist[idx][0], x_limits_dist[idx][1], 100))
    y_dist.append(skewnorm.pdf(x_dist[idx], norm_param[idx][0], norm_param[idx][1], norm_param[idx][2]))
    ax.plot(x_dist[idx], y_dist[idx])
    ax.set_title(titles[idx])
    ax.set_xlabel("%s Tempo" % titles[idx])
    ax.set_ylabel("Density")
    ax.grid()

plt.show()

# plot distributions together
for idx, dist in enumerate(y_dist):
    plt.plot(x_dist[idx], dist)
    plt.xlabel("Tempo")
    plt.ylabel("Density")
    plt.grid()

plt.show()

# plot histogram of K-fold scores
plt.hist(scores, folds, facecolor="#0000FF")
plt.title("Density of scores of K-fold cross-validation")
plt.xlabel("Scores")
plt.ylabel("Density")
plt.grid()
plt.show()