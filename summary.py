from auxiliary import load_csv, one_out_of_k
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show, grid
from scipy.linalg import svd
import numpy as np

attribute_names, raw_data = load_csv("./res/spotify-data-apr-2019.csv")
class_names, X = one_out_of_k(raw_data, column_index=13, return_uniques=True)
class_dict = dict(zip(range(len(class_names)), class_names))

y = raw_data[:, 13]
N = len(y)
M = len(attribute_names)
C = len(class_names)

i = 3  # energy column
j = 7  # loudness column

###############################################
# Age vs Bone-Density plot
###############################################

f1 = figure()
title('Spotify API Data')

for color in range(C):
    # select indices belonging to class c:
    class_mask = y == class_dict[color]
    q = raw_data[class_mask, i]
    if color == 4:
        plot(raw_data[class_mask, i], raw_data[class_mask, j], 'o', alpha=1, markevery=20)
    else:
        plot(raw_data[class_mask, i], raw_data[class_mask, j], 'o', alpha=1, markevery=150)

legend(class_names)
xlabel(attribute_names[i])
ylabel(attribute_names[j])
show()

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
plot(range(1, len(rho) + 1), rho, 'x-')
plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
plot([1, len(rho)], [threshold, threshold], 'k--')
title('Variance explained by principal components')
xlabel('Principal component')
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
title('NanoNose data: PCA')
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
