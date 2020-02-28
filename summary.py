from auxiliary import one_out_of_k
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show, grid, xticks, yticks
from scipy.linalg import svd
import numpy as np
import seaborn as sb
import pandas as pd

file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)
attribute_names = df_data.columns.values
data = df_data.values
class_names, X = one_out_of_k(data, column_index=13, return_uniques=True)
class_dict = dict(zip(range(len(class_names)), class_names))

y = data[:, 13]
N = len(y)
M = len(attribute_names)
C = len(class_names)

i = 3  # energy column
j = 7  # loudness column

###############################################
# Seaborn Plotting
###############################################

palette = ['blue', 'purple', 'red', 'green', 'black']

# print plot of energy vs loudness
sb.set(style="ticks", rc={'figure.figsize':(16,6)})
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
f2 = figure()
xl = [sum(x) for x in zip(list(range(len(rho))), [1]*len(rho))]
plot(range(1, len(rho) + 1), rho, 'x-')
plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
plot([1, len(rho)], [threshold, threshold], 'k--')
title('Variance explained by principal components')
xlabel('Principal component')
xticks(xl, xl)
ylabel('Variance explained')
legend(['Individual', 'Cumulative', 'Threshold'])
grid()
show()

###############################################
# Principal Component Analysis Algorithm
###############################################

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = V.T

# Project the centered data onto principal component space
Z = Y @ V

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
