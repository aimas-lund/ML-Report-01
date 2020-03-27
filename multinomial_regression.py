# exercise 8.3.2 Fit multinomial regression
from matplotlib.pyplot import figure, show, title
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import numpy as np
import sklearn.linear_model as lm

# Load Matlab data file and extract variables of interest

"""
mat_data = loadmat('../Data/synth1.mat')
X = mat_data['X']
X = X - np.ones((X.shape[0],1)) * np.mean(X,0)
X_train = mat_data['X_train']
X_test = mat_data['X_test']
y = mat_data['y'].squeeze()
y_train = mat_data['y_train'].squeeze()
y_test = mat_data['y_test'].squeeze()

attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]

N, M = X.shape
C = len(classNames)
"""
from auxiliary import one_out_of_k, add_elements_to_list
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, title
from scipy import stats
from scipy.linalg import svd
import numpy as np
import seaborn as sb
import pandas as pd
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
from sklearn.model_selection import train_test_split

def every_nth(input, n, iteration=1):
    output = input

    for i in range(iteration):
        output = output[np.mod(np.arange(output.size), n) != 0]

    return output

def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)
attribute_names = df_data.columns.values
data = df_data.values


y = data[:,[13]]             # Tempo(target)
selector = [x for x in range(data.shape[1]) if x != 13]
X = data[:, selector] # the rest of the data set
#X = data[:,:14]           # the rest of features

attributeNames = ['acousticness', 'danceability', 'duration_ms', 'energy',
                  'instrumentalness', 'key','liveness', 'loudness', 'mode',
                   'speechiness', 'time_signature', 'valence', 'popularity_interval']
################################################
#Neuralnetwork classification
################################################

# Load Matlab data file and extract variables of interest

X = X - np.ones((X.shape[0],1)) * np.mean(X,0)


# Simple holdout-set crossvalidation
vf = 0.2                                    # fraction of the data reserved for validation (usually between 0.2 and 0.3)
# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)



classNames = ['Popularity interval 1', 'Popularity interval 2', 'Popularity interval 3',
               'Popularity interval 4', 'Popularity interval 5']

N, M = X.shape
C = len(classNames)
#%% Model fitting and prediction

# Multinomial logistic regression
logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=1)
logreg.fit(X_train,y_train)

# To display coefficients use print(logreg.coef_). For a 4 class problem with a 
# feature space, these weights will have shape (4, 2).

# Number of miss-classifications
print('Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}'.format(np.sum(logreg.predict(X_test)!=y_test),len(y_test)))

predict = lambda x: np.argmax(logreg.predict_proba(x),1)
figure(2,figsize=(9,9))
visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
title('LogReg decision boundaries')

show()

print('Ran Exercise 8.3.2')
