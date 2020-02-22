from auxiliary import load_csv, one_out_of_k
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show, grid
from scipy.linalg import svd
import numpy as np

attribute_names, raw_data = load_csv("./res/bone_density.csv")
class_names, X = one_out_of_k(raw_data, column_index=1, return_uniques=True)
class_dict = dict(zip(range(len(class_names)), class_names))

y = raw_data[:, 1]
N = len(y)
M = len(attribute_names)
C = len(class_names)

i = 2  # age column
j = 4  # bone density column

###############################################
# Age vs Bone-Density plot
###############################################

f1 = figure()
title('Age vs Bone-density')

for color in range(C):
    # select indices belonging to class c:
    class_mask = y == class_dict[color]
    q = raw_data[class_mask, i]
    plot(raw_data[class_mask, i], raw_data[class_mask, j], 'o', alpha=.8)

legend(class_names)
xlabel(attribute_names[i])
ylabel(attribute_names[j])
show()

###############################################
# Principal Component Analysis
###############################################
X = X[:,1:]
Y = np.array(X - np.ones((N, 1)) * X.mean(axis=0), dtype=float)

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
