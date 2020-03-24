from auxiliary import one_out_of_k, every_nth
import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np
import seaborn as sb
import pandas as pd


file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)            # data as pandas DataFrame format
attribute_names = df_data.columns.values    # numpy array of attribute names
data = df_data.values                       # all data in numpy array format

class_names, X = one_out_of_k(data, column_index=13, return_uniques=True)  # One-out-of-K on 'popularity_interval'
class_dict = dict(zip(range(len(class_names)), class_names))

y = data[:, 13]             # class belonging to each row in normal format
y_k = X[:, 13:]             # class attributes with one-out-of-k encoding
X = X[:, :13]               # data set with all non-class attributes
N = len(y)                  # number of observations
M = len(attribute_names)    # number of attributes (including class attributes)
C = len(class_names)        # number of class attributes

###############################################
# Summary Statistics
###############################################

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

###############################################
# Remaining boxplots
###############################################


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

###############################################
# Principal Component Analysis
###############################################

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
#Z = np.array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c+1
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', 
             alpha=.5, 
             c=color[c])
plt.legend(class_names)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()

i = 6
j = 8
"""
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
"""

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

######################################
#Histogram
######################################

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
#1 3 6 11

df = pd.DataFrame(X)
plt.matshow(df.corr())
plt.colorbar()
plt.show()

"""
for i in range(M-C):
    for j in range(M-C):
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

###############################################
# Principal Component Analysis Algorithm
###############################################


pca_names = []
coeffs = []
for i in range(len(rho)):
    pca_names.append("PC{}".format(i + 1))
for i in range(len(rho)):
    coeffs.append("c{}".format(i + 1))
pca_df = pd.DataFrame(data=V, columns=pca_names)

# print first 9 PC's

fig_pca, axs_pca = plt.subplots(3, 3, figsize=(16, 8), constrained_layout=True)
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

fig_pca, axs_pca = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
plt.setp(axs_pca,
         xticks=np.arange(len(pca_names)),
         xticklabels=coeffs,
         yticks=[-1.0, -.75, -.5, -.25, 0.0, .25, .5, .75, 1.0])

for i in range(0, 4):
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
axs_pca.set_title(pca_names[12] + " Coefficients")
axs_pca.bar(np.arange(len(pca_names)), V[12])
axs_pca.axhline(linewidth=1, color='black')
axs_pca.set_xticklabels(coeffs, rotation=45)

plt.show()
