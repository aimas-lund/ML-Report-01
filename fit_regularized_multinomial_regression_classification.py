import sklearn.linear_model as lm
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
N, M = X.shape
C = 2
attributeNames = ['acousticness', 'danceability', 'duration_ms', 'energy',
                  'instrumentalness', 'key','liveness', 'loudness', 'mode',
                   'speechiness', 'time_signature', 'valence', 'popularity_interval']

# Load Matlab data file and extract variables of interest

X = X - np.ones((X.shape[0],1)) * np.mean(X,0)


# Simple holdout-set crossvalidation
vf = 0.2                                    # fraction of the data reserved for validation (usually between 0.2 and 0.3)
# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)
y_test = np.squeeze(y_test) -1
y_train = np.squeeze(y_train) -1
y = np.squeeze(y) -1


classNames = np.squeeze(np.array(['Popularity interval 1', 'Popularity interval 2', 'Popularity interval 3',
               'Popularity interval 4', 'Popularity interval 5']))

N, M = X.shape
C = len(classNames)
#%% Model fitting and prediction
# Standardize data based on training set
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Fit multinomial logistic regression model
#regularization_strength = 1e-3
regularization_strength = 1e5
#Try a high strength, e.g. 1e5, especially for synth2, synth3 and synth4
mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                               tol=1e-4, random_state=1, 
                               penalty='l2', C=1/regularization_strength)
mdl.fit(X_train,y_train)
y_test_est = mdl.predict(X_test)

test_error_rate = np.sum(y_test_est!=y_test) / len(y_test)

predict = lambda x: np.argmax(mdl.predict_proba(x),1)
#plt.figure(2,figsize=(9,9))
#visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
#plt.title('LogReg decision boundaries')
#plt.show()


# Number of miss-classifications
print('Error rate: \n\t {0} % out of {1}'.format(test_error_rate*100,len(y_test)))
# %%

#plt.figure(2, figsize=(9,9))
#plt.hist([y_train, y_test, y_test_est], color=['red','green','blue'], density=True)
#plt.legend(['Training labels','Test labels','Estimated test labels'])


print('Ran Exercise 8.3.2')