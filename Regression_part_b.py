

from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from toolbox_02450 import jeffrey_interval
from toolbox_02450 import mcnemar
from math import sqrt

from sklearn import linear_model


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)

reg = linear_model.Ridge(alpha=0.1) #alpha is the same as lambda
reg.fit(X_train,y_train)

#y_estimated = model.predict(y_test)
#residual = y_estimated-y_test
#print("Mean squared error: %.2f" % mean_squared_error(y_test,y_estimated))
#print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test,y_estimated)))


model = lm.LinearRegression()
model = model.fit(X_train,y_train)

predictions = cross_val_predict(model, X_test, y_test, cv=folds)  # calculates the test accuracy

print("Mean squared error: %.2f" % mean_squared_error(y_test,predictions))
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test,predictions)))
"""
# requires data from exercise 1.5.1
from ex1_5_1 import *

# This script crates predictions from three KNN classifiers using cross-validation

# Maximum number of neighbors
L=[1, 20, 80]

CV = model_selection.LeaveOneOut()
i=0

# store predictions.
yhat = []
y_true = []
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    dy = []
    for l in L:
        knclassifier = KNeighborsClassifier(n_neighbors=l)
        knclassifier.fit(X_train, y_train)
        y_est = knclassifier.predict(X_test)

        dy.append( y_est )
        # errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    i+=1

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)
yhat[:,0] # predictions made by first classifier.
# Compute accuracy here.

# Compute the Jeffreys interval
alpha = 0.05
[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)



# Compute the Mcnemar interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)
"""