from auxiliary import one_out_of_k, add_elements_to_list
import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np
import seaborn as sb
import pandas as pd


def every_nth(input, n, iteration=1):
    output = input

    for i in range(iteration):
        output = output[np.mod(np.arange(output.size), n) != 0]

    return output


file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)
attribute_names = df_data.columns.values
data = df_data.values

class_names, X = one_out_of_k(data, column_index=13, return_uniques=True)  # One-out-of-K on 'popularity_interval'
class_dict = dict(zip(range(len(class_names)), class_names))

y = data[:, 13] # class belonging to each row in normal format
y_3ok = X[:, 13:]
X = X[:, :13]
N = len(y)
M = len(attribute_names)
C = len(class_names)

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

###############################################
# Seaborn Plotting
###############################################

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

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = V.T

# Project the centered data onto principal component space
Z = Y @ V

pca_names = []
coeffs = []
for i in range(len(rho)):
    pca_names.append("PC{}".format(i + 1))
for i in range(len(rho)):
    coeffs.append("c{}".format(i + 1))
pca_df = pd.DataFrame(data=V, columns=pca_names)


def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


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

fig_pca, axs_pca = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
axs_pca = trim_axs(axs_pca, pca_df.shape[1])
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
