from auxiliary import one_out_of_k, add_elements_to_list
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import svd
import numpy as np
import seaborn as sb
import pandas as pd


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

class_names, X = one_out_of_k(data, column_index=13, return_uniques=True)  # One-out-of-K on 'popularity_interval'
mode_names, X = one_out_of_k(X, column_index=8, return_uniques=True)  # One-out-of-K on 'mode'
class_dict = dict(zip(range(len(class_names)), class_names))

y = data[:, 13]
N = len(y)
M = len(attribute_names)
C = len(class_names)

###############################################
# Summary Statistics
###############################################
"""
data_mean = data.mean(axis=0)
data_std = data.std(axis=0)
data_min = data.min(axis=0)
data_max = data.max(axis=0)

box_data = df_data.drop(['mode', 'key', 'loudness', 'tempo', 'time_signature', 'popularity_interval', 'duration_ms'],
                        axis=1)

f, ax = plt.subplots()
ax = sb.boxplot(data=box_data, showfliers=False)
plt.xticks(rotation=-15)
plt.show()

plt.boxplot(box_data.values, sym='+')
plt.xticks(range(1, 8), box_data.columns.values)
plt.show()
"""
###############################################
# Remaining boxplots
###############################################

"""
remainder = ['duration_ms', 'tempo', 'loudness']
box_ylabels = ['Duration (ms)', 'Tempo (bps)', 'Loudness (dB)']

fig, axs = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)

for i in range(len(remainder)):
    axs[0, i].boxplot(df_data[remainder[i]], showfliers=False)
    axs[0, i].set_ylabel(box_ylabels[i])
    axs[0, i].set_xticks([])

for i in range(len(remainder)):
    axs[1, i].boxplot(every_nth(df_data[remainder[i]].values, 2, 2), sym='+')
    axs[1, i].set_ylabel(box_ylabels[i])
    axs[1, i].set_xticks([])

plt.show()
"""
###############################################
# Seaborn Plotting
###############################################
"""
palette = ['blue', 'purple', 'red', 'green', 'black']
# print plot of energy vs loudness
sb.set(style="ticks", rc={'figure.figsize': (16, 6)})
sb.relplot(x="energy", y="loudness",
           hue="popularity_interval",
           palette=palette,
           col="popularity_interval",
           alpha=0.5,
           data=df_data)
"""
###############################################
# Principal Component Analysis
###############################################
"""
Y = np.array(X - np.ones((N, 1)) * X.mean(axis=0), dtype=float) / X.std(axis=0)

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = V.T

# Project the centered data onto principal component space
Z = Y @ V
# Indices of the principal components to be plotted

i = 3
j = 7

# Plot PCA of the data
f = plt.figure()
plt.title('PCA')
color = np.array(['b','m','r','g','k'])
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c+1
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', 
             alpha=.5, 
             c=color[class_mask])
plt.legend(class_names)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()

i = 6
j = 8

# Plot PCA of the data
f = plt.figure()
plt.title('PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c+1
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', 
             alpha=.5, 
             c=np.array(['w','b','m','r','g','k']))
plt.legend(class_names)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()


i = 3
j = 7
f = plt.figure()
plt.title('PCA')

#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c+1
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', 
             alpha=.5)
plt.legend(class_names)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()
i = 6
j = 8

# Plot PCA of the data
f = plt.figure()
plt.title('PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c+1
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', 
             alpha=.5)
plt.legend(class_names)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()

from scipy import stats
"""
######################################
#Histogram
######################################
"""
# Number of bins in histogram
nbins = 20

# Plot the histogram
f = plt.figure()
plt.title('Normal distribution')
plt.hist(X[:,0], bins=nbins, density=False)

# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,0].min(), X[:,0].max(), 1000)
pdf = stats.norm.pdf(x,loc=17,scale=2)
plt.plot(x,pdf,'.',color='red')

#hist = sb.distplot(X[:,2],norm_hist=True)
plt.hist(X[:, 1], bins=200)
plt.title('Danceability')
plt.ylabel('Frequency')
plt.xlabel('Value')
plt.show()
"""
#1 3 6 11
"""
#Covariance matrix
df = pd.DataFrame(X)
plt.matshow(df.corr())
plt.colorbar()
plt.show()


for i in range(19):
    for j in range(19):
# Plot PCA of the data
        f = plt.figure()
        plt.title('PCA')
        #Z = array(Z)
        for c in range(C):
            # select indices belonging to class c:
            class_mask = y==c+1
            plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
        plt.legend(class_names)
        plt.xlabel('PC{0}'.format(i+1))
        plt.ylabel('PC{0}'.format(j+1))
        
        # Output result to screen
        plt.show()

"""
"""
threshold = 0.9

# Plot variance explained
f2 = plt.figure()
xl = [sum(x) for x in zip(list(range(len(rho))), [1] * len(rho))]
plt.plot(range(1, len(rho) + 1), rho, 'x-')
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.xticks(xl, xl)
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()
plt.show()
"""
###############################################
# Principal Component Analysis Algorithm
###############################################
"""

pca_names = []
coeffs = []
for i in range(len(rho)):
    pca_names.append("PC{}".format(i + 1))
for i in range(len(rho)):
    coeffs.append("c{}".format(i + 1))
pca_df = pd.DataFrame(data=V, columns=pca_names)

# print first 9 PC's

fig_pca, axs_pca = plt.subplots(3, 3, figsize=(16, 8), constrained_layout=True)
axs_pca = trim_axs(axs_pca, pca_df.shape[1])
plt.setp(axs_pca,
         xticks=np.arange(len(pca_names)),
         xticklabels=coeffs,
         yticks=[-1.0, -.75, -.5, -.25, 0.0, .25, .5, .75, 1.0])

for i in range(0, 9):
    axs_pca[i].yaxis.grid(color='gray', linestyle='dashed')
    axs_pca[i].set_title(pca_names[i] + " Coefficients")
    axs_pca[i].bar(np.arange(len(pca_names)), V[i])
    axs_pca[i].axhline(linewidth=1, color='black')
    axs_pca[i].set_xticklabels(coeffs, rotation=45)

plt.show()

# show PC's for index 9-18

fig_pca, axs_pca = plt.subplots(3, 3, figsize=(16, 8), constrained_layout=True)
axs_pca = trim_axs(axs_pca, pca_df.shape[1])
plt.setp(axs_pca,
         xticks=np.arange(len(pca_names)),
         xticklabels=coeffs,
         yticks=[-1.0, -.75, -.5, -.25, 0.0, .25, .5, .75, 1.0])

for i in range(0, 9):
    axs_pca[i].yaxis.grid(color='gray', linestyle='dashed')
    axs_pca[i].set_title(pca_names[i+9] + " Coefficients")
    axs_pca[i].bar(np.arange(len(pca_names)), V[i+9])
    axs_pca[i].axhline(linewidth=1, color='black')
    axs_pca[i].set_xticklabels(coeffs, rotation=45)

plt.show()

# show coefficients for last PC
fig_pca, axs_pca = plt.subplots(1, 1, figsize=(5, 2), constrained_layout=True)
plt.setp(axs_pca,
         xticks=np.arange(len(pca_names)),
         xticklabels=coeffs,
         yticks=[-1.0, -.75, -.5, -.25, 0.0, .25, .5, .75, 1.0])

axs_pca.yaxis.grid(color='gray', linestyle='dashed')
axs_pca.set_title(pca_names[18] + " Coefficients")
axs_pca.bar(np.arange(len(pca_names)), V[18])
axs_pca.axhline(linewidth=1, color='black')
axs_pca.set_xticklabels(coeffs, rotation=45)

plt.show()
"""
################################################
# Linear regression
################################################
"""

from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm


# Split dataset into features and target vector
tempo_idx = 11
y = X[:,tempo_idx]

X_cols = list(range(0,tempo_idx)) + list(range(tempo_idx+1,19))
X = X[:,X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Tempo (true)'); ylabel('Tempo (estimated)');
subplot(2,1,2)
hist(residual,40)

show()
"""
################################################
#Neuralnetwork tempo
################################################
# exercise 8.2.6
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats


y = data[:,[10]]             # Tempo(target)
selector = [x for x in range(data.shape[1]) if x != 10]
X = data[:, selector] # the rest of the data set
#X = data[:,:14]           # the rest of features
N, M = X.shape
C = 2
attributeNames = ['acousticness', 'danceability', 'duration_ms', 'energy',
                  'instrumentalness', 'key','liveness', 'loudness', 'mode',
                   'speechiness', 'time_signature', 'valence', 'popularity_interval']
# Normalize data
X = stats.zscore(X)
                
## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(X,0)
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 3
    X = X @ V[:,:k_pca]
    N, M = X.shape


# Parameters for neural network classifier
n_hidden_units = 20     # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

# K-fold crossvalidation
K = 5                  # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue',
              'tab:orange','tab:green', 'tab:purple']
# Define the model
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold 
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Test mean-squared-error')
    
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Tempo: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value (bmp)')
plt.ylabel('Estimated value (bpm)')
plt.grid()

plt.show()
