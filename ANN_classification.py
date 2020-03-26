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
################################################
#Neuralnetwork classification
################################################

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

# Define the model structure
n_hidden_units = 5 # number of hidden units in the signle hidden layer
model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            # Output layer:
                            # H hidden units to C classes
                            # the nodes and their activation before the transfer 
                            # function is often referred to as logits/logit output
                            torch.nn.Linear(n_hidden_units, C), # C logits
                            # To obtain normalised "probabilities" of each class
                            # we use the softmax-funtion along the "class" dimension
                            # (i.e. not the dimension describing observations)
                            torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output
                            )
# Since we're training a multiclass problem, we cannot use binary cross entropy,
# but instead use the general cross entropy loss:
loss_fn = torch.nn.CrossEntropyLoss()
# Train the network:
net, _, _ = train_neural_net(model, loss_fn,
                             X=torch.tensor(X_train, dtype=torch.float),
                             y=torch.tensor(y_train, dtype=torch.long),
                             n_replicates=3,
                             max_iter=10000)
# Determine probability of each class using trained network
softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
# Get the estimated class as the class with highest probability (argmax on softmax_logits)
y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
# Determine errors
e = (y_test_est != y_test)
print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))

predict = lambda x:  (torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]).data.numpy() 
# figure(1,figsize=(9,9))
# visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
# title('ANN decision boundaries')

# show()
