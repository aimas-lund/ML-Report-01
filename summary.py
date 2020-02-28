from auxiliary import one_out_of_k, add_elements_to_list
import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np
import seaborn as sb
import pandas as pd

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

plt.figure(figsize=(9, 3))

plt.subplot(131)
sb.boxplot(data=df_data['duration_ms'], showfliers=False, orient='h')
plt.xlabel("Duration (ms)")
plt.subplot(132)
sb.boxplot(data=df_data['tempo'], showfliers=False, color='red', orient='h')
plt.xlabel("Tempo (BPS)")
plt.subplot(133)
sb.boxplot(data=df_data['loudness'], showfliers=False, color='purple', orient='h')
plt.xlabel("Loudness (dB)")
plt.suptitle('Categorical Plotting')
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

"""
# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Spotify Music Attributes: PCA')
# Z = array(Z)
for color in range(C):
    # select indices belonging to class c:
    class_mask = y == class_dict[color]
    if color == 4:
        plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=1, markevery=20)
    else:
        plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.8, markevery=150)
legend(class_names)
xlabel('PC{0}'.format(i + 1))
ylabel('PC{0}'.format(j + 1))

# Output result to screen
show()

# Reformat attribute names for transformed matrix
attribute_names = add_elements_to_list(attribute_names,
                                       class_names,
                                       len(attribute_names),
                                       added_string='popularity_interval ')
attribute_names = add_elements_to_list(attribute_names,
                                       mode_names,
                                       8,
                                       added_string='mode ')
attribute_names = np.delete(attribute_names,
                            np.where(attribute_names == 'mode'))
attribute_names = np.delete(attribute_names,
                            np.where(attribute_names == 'popularity_interval'))
"""
pca_names = []
coeffs = []
for i in range(len(rho)):
    pca_names.append("PCA{}".format(i + 1))
for i in range(len(rho)):
    coeffs.append("c{}".format(i + 1))
pca_df = pd.DataFrame(data=V, columns=pca_names)

for i in range(len(pca_names)):
    f, ax = plt.subplots()
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.xticks(np.arange(len(pca_names) + 2), coeffs, rotation=-20)
    plt.title(pca_names[i] + " Coefficients")
    plt.axhline(linewidth=1, color='black')
    plt.bar(np.arange(len(pca_names)), V[i])
    plt.show()
