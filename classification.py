from auxiliary import one_out_of_k
from scipy.linalg import svd
import numpy as np
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