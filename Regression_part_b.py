import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from toolbox_02450 import jeffrey_interval
from toolbox_02450 import mcnemar
from math import sqrt
from sklearn.metrics import mean_squared_error
import sklearn.linear_model as lm 
from sklearn import linear_model
import copy

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
folds = 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf, shuffle =False)


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso


import numpy as np
from matplotlib import pyplot as plt
"""
n_alphas = 200
alphas = np.logspace(1, 4, n_alphas)


folds = 5
RMSE = []
for a in alphas:
    model = linear_model.Ridge(alpha=a, fit_intercept = False)
    predictions = cross_val_predict(model, X, y, cv=folds) 
    RMSE.append(sqrt(mean_squared_error(y,predictions)))

index = RMSE.index(min(RMSE))
print ("RMSE min :" ,RMSE[index])
print ("lambda value at min RMSE :" ,alphas[index])
plt.figure(figsize=(5, 3))

plt.plot(alphas, RMSE)
plt.xscale("log")
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()
"""

"""
n_alphas = 10
alphas = np.logspace(2, 4, n_alphas)


folds = 5
MSE = []
for a in alphas:
    model = linear_model.Ridge(alpha=a, fit_intercept = False)
    predictions = cross_val_predict(model, X, y, cv=folds) 
    MSE.append(mean_squared_error(y,predictions))

index = MSE.index(min(MSE))
print ("RMSE min :" ,MSE[index])
print ("lambda value at min RMSE :" ,alphas[index])
plt.figure(figsize=(5, 3))

plt.plot(alphas, MSE)
plt.xscale("log")
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()

print(MSE)
"""


# Fit ordinary least squares regression model
vf = 0.2
folds = 5 # fold for k-folds

# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf, shuffle =False)

mean = int(np.mean(y_train))
y_train_mean = copy.copy(y_train)
y_train_mean.fill(mean)

y_test_mean = copy.copy(y_test)
y_test_mean.fill(mean)
#y_estimated = model.predict(y_test)
#residual = y_estimated-y_test

model_base = lm.LinearRegression()
model_base = model_base.fit(y_train_mean,y_train)

model_reg = linear_model.Ridge(alpha=278.25, fit_intercept = False)
model_reg = model_reg.fit(X_train,y_train)


import scipy.stats
import numpy as np, scipy.stats as st

yhat_base = model_base.predict(y_test_mean)
yhat_reg = model_reg.predict(X_test)

y_est_ann = np.loadtxt("y_est.csv", delimiter=",")
y_true_ann = np.loadtxt("y_true.csv", delimiter=",")

z_base = np.abs(y_test - yhat_base ) ** 2
z_reg = np.abs(y_test - yhat_reg ) ** 2
z_ann = np.abs(y_true_ann - y_est_ann) ** 2

# compute confidence interval 
alpha = 0.05
CI_base = st.t.interval(1-alpha, df=len(z_base)-1, loc=np.mean(z_base), scale=st.sem(z_base))  # Confidence interval
CI_reg = st.t.interval(1-alpha, df=len(z_reg)-1, loc=np.mean(z_reg), scale=st.sem(z_reg))  # Confidence interval
CI_ann = st.t.interval(1-alpha, df=len(z_ann)-1, loc=np.mean(z_ann), scale=st.sem(z_ann))  # Confidence interval

z_reg_base = z_reg - z_base
z_ann_reg = z_ann - z_reg
z_ann_base = z_ann - z_base

CI_reg_base = st.t.interval(1-alpha, len(z_reg_base)-1, loc=np.mean(z_reg_base), scale=st.sem(z_reg_base))  # Confidence interval
CI_ann_reg = st.t.interval(1-alpha, len(z_ann_reg)-1, loc=np.mean(z_ann_reg), scale=st.sem(z_ann_reg))  # Confidence interval
CI_ann_base = st.t.interval(1-alpha, len(z_ann_base)-1, loc=np.mean(z_ann_base), scale=st.sem(z_ann_base))  # Confidence interval

p_reg_base = st.t.cdf( -np.abs( np.mean(z_reg_base) )/st.sem(z_reg_base), df=len(z_reg_base)-1)  # p-value
p_ann_reg = st.t.cdf( -np.abs( np.mean(z_ann_reg) )/st.sem(z_ann_reg), df=len(z_ann_reg)-1)  # p-value
p_ann_base = st.t.cdf( -np.abs( np.mean(z_ann_base) )/st.sem(z_ann_base), df=len(z_ann_base)-1)  # p-value

print("finished")