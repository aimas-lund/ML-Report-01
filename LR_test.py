# exercise 5.2.4
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm




from auxiliary import one_out_of_k, add_elements_to_list
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import svd
import numpy as np
import seaborn as sb
import pandas as pd
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error

file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)
attribute_names = df_data.columns.values
data = df_data.values

y = data[:,[10]]             # Tempo(target)
selector = [x for x in range(data.shape[1]) if x != 10]
X = data[:, selector] # the rest of the data set
#X = data[:,:14]           # the rest of features

# Fit ordinary least squares regression model
vf = 0.2
folds = 5 # fold for k-folds

# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf, shuffle =False)

mean = int(np.mean(y_train))
y_train_mean = y_train
y_train_mean.fill(mean)


model = lm.LinearRegression()
model = model.fit(y_train_mean,y_train)

y_test_mean=y_test
y_test_mean.fill(mean)
#y_estimated = model.predict(y_test)
#residual = y_estimated-y_test

predictions = cross_val_predict(lin_model, y_test_mean, y_test, cv=folds)  # calculates the test accuracy

print("Mean squared error: %.2f" % mean_squared_error(y_test,predictions))
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test,predictions)))



"""
# Display scatter plot
figure()
subplot(2,1,1)
plot(y_test, y_est, '.')
xlabel('Tempo (true)'); ylabel('Tempo (estimated)');
subplot(2,1,2)
hist(residual,40)

show()

print('Ran Exercise 5.2.4')
"""