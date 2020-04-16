import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from scipy.stats import skewnorm, norm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from auxiliary import one_out_of_k, calc_distribution, trim_ticks
import math
from sklearn.metrics import mean_squared_error


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
X = one_out_of_k(X, 12)  # one out of K on popularity interval

#X = preprocessing.scale(X)
folds = 10  # fold for k-folds x-validation

# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)
"""
lin_model = lm.LinearRegression()  # create model

# the test set we keep for testing
# the training set we use k-folds cross validation
scores = cross_val_score(lin_model, X_train, y_train, cv=folds)  # calculates x-validation scores
print("X-validation used: %i-folds" % folds)
print("X-validation accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

predictions = cross_val_predict(lin_model, X_test, y_test, cv=folds)  # calculates the test accuracy
accuracy = metrics.r2_score(y_test, predictions)  # calculates the R^2 value between true- & predicted values
print("Test Accuracy: %0.5f\n" % accuracy)

tempo_corr = df_data.corr().iloc[10].drop(['popularity_interval', 'tempo'])
plt.figure(figsize=(5, 3))
plt.bar(trim_ticks(tempo_corr.index.values), tempo_corr.values, color="#0000FF")
plt.xticks(np.arange(12), rotation=75)
plt.ylabel("Correlation Coefficient")
plt.grid()
# plt.savefig("correlation.png",bbox_inches='tight',dpi=100)
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
    x_limits_dist.append(get_percentiles(data, 0.01, 99.99))

for idx, ax in enumerate(axs):
    ax.hist(tempos[idx], bins=30, density=True, facecolor="#0000FF")  # plot histogram with densities
    # calculate the x- and y-values of the fitted, skewed normal distribution
    x_dist.append(np.linspace(x_limits_dist[idx][0], x_limits_dist[idx][1], 100))
    y_dist.append(skewnorm.pdf(x_dist[idx], norm_param[idx][0], norm_param[idx][1], norm_param[idx][2]))
    ax.plot(x_dist[idx], y_dist[idx])  # plot distributions
    ax.set_title(titles[idx])
    ax.set_xlabel("%s Tempo" % titles[idx])
    ax.set_ylabel("Density")
    ax.grid()

plt.show()

# plot distributions together
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
color = ['#4C4C4C', '#0000FF']
for idx, dist in enumerate(y_dist):
    ax[0].plot(x_dist[idx], dist, c=color[idx])
    ax[0].set_xlabel("Tempo")
    ax[0].set_xlim(x_limits_dist[1])
    ax[0].set_ylabel("Density")
    ax[0].legend(titles)
    ax[0].grid()

residuals = predictions - y_test

d_res_x, d_res_y = calc_distribution(residuals)

ax[1].hist(residuals, bins=40, density=True, facecolor="#0000FF")
ax[1].plot(d_res_x, d_res_y)
ax[1].set_xlabel("Tempo estimation error")
ax[1].set_ylabel("Density")
ax[1].grid()
plt.show()

# plot histogram of K-fold scores
plt.hist(scores, folds, facecolor="#0000FF")
plt.title("Density of scores of K-fold cross-validation")
plt.xlabel("Cross-validation score")
plt.ylabel("Count")
plt.grid()
plt.show()

# Try only coefficients with highest covariance with the tempo attribute
df_data = df_data[['acousticness', 'energy', "loudness", "tempo"]]  # data as pandas DataFrame format
df_data.sample(frac=1)  # shuffle rows of data
data = df_data.values  # all data in numpy array format

vf = 0.2  # fraction of the data reserved for validation (usually between 0.2 and 0.3)
N = data.shape[0]  # number of rows in the data set
M = data.shape[1]  # number of columns in the data set
y = data[:, 3]  # class belonging to each row in normal format
X = np.delete(data, 3, axis=1)
X = preprocessing.scale(X)
folds = 10  # fold for k-folds x-validation

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
    x_limits_dist.append(get_percentiles(data, 0.01, 99.99))

for idx, ax in enumerate(axs):
    ax.hist(tempos[idx], bins=30, density=True, facecolor="#0000FF")  # plot histogram with densities
    # calculate the x- and y-values of the fitted, skewed normal distribution
    x_dist.append(np.linspace(x_limits_dist[idx][0], x_limits_dist[idx][1], 100))
    y_dist.append(skewnorm.pdf(x_dist[idx], norm_param[idx][0], norm_param[idx][1], norm_param[idx][2]))
    ax.plot(x_dist[idx], y_dist[idx])  # plot distributions
    ax.set_title(titles[idx])
    ax.set_xlabel("%s Tempo" % titles[idx])
    ax.set_ylabel("Density")
    ax.grid()

plt.show()

# plot distributions together
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
color = ['#4C4C4C', '#0000FF']
for idx, dist in enumerate(y_dist):
    ax[0].plot(x_dist[idx], dist, c=color[idx])
    ax[0].set_xlabel("Tempo")
    ax[0].set_xlim(x_limits_dist[1])
    ax[0].set_ylabel("Density")
    ax[0].legend(titles)
    ax[0].grid()

residuals = predictions - y_test

d_res_x, d_res_y = calc_distribution(residuals)

ax[1].hist(residuals, bins=40, density=True, facecolor="#0000FF")
ax[1].plot(d_res_x, d_res_y)
ax[1].set_xlabel("Tempo estimation error")
ax[1].set_ylabel("Density")
ax[1].grid()
plt.show()

# plot histogram of K-fold scores
plt.hist(scores, folds, facecolor="#0000FF")
plt.title("Density of scores of K-fold cross-validation")
plt.xlabel("Cross-validation score")
plt.ylabel("Count")
plt.grid()
plt.show()
"""
# plot RMSE as a function of regularization rate for Lasso and Ridge
n_alphas = 150
alphas = np.logspace(0, 4, n_alphas)

RMSE_1 = []

for a in alphas:
    model = lm.Ridge(alpha=a, fit_intercept=False)
    predictions = cross_val_predict(model, X, y, cv=folds)
    RMSE_1.append(math.sqrt(mean_squared_error(y, predictions)))

index = RMSE_1.index(min(RMSE_1))
print("RMSE min :", RMSE_1[index])
print("lambda value at min RMSE :", alphas[index])

color = ['#4C4C4C', '#0000FF']

plt.figure(figsize=(5, 3))
plt.plot(alphas, RMSE_1, c=color[1])
plt.xscale("log")
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.grid()
plt.tight_layout()
plt.show()
