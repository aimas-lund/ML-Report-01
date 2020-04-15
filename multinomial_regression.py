# exercise 8.3.2 Fit multinomial regression
import sklearn.linear_model as lm
from auxiliary import one_out_of_k, add_elements_to_list
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import recall_score, precision_score, precision_recall_curve

file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)
attribute_names = df_data.columns.values
data = df_data.values

vf = 0.2  # fraction of the data reserved for validation (usually between 0.2 and 0.3)
N = data.shape[0]  # number of rows in the data set
M = data.shape[1]  # number of columns in the data set
y = data[:, 13]  # class belonging to each row in normal format
X = np.delete(data, 13, axis=1)
folds = 10  # fold for k-folds x-validation

attributeNames = ['acousticness', 'danceability', 'duration_ms', 'energy',
                  'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                  'speechiness', 'time_signature', 'valence', 'popularity_interval']

rus = RandomUnderSampler(random_state=0)
X, y = rus.fit_resample(X, y)

# Simple holdout-set crossvalidation
vf = 0.2  # fraction of the data reserved for validation (usually between 0.2 and 0.3)
# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)
y_test = np.squeeze(y_test) - 1
y_train = np.squeeze(y_train) - 1
y = np.squeeze(y) - 1

classNames = ['Popularity interval 1', 'Popularity interval 2', 'Popularity interval 3',
              'Popularity interval 4', 'Popularity interval 5']

N, M = X.shape
C = len(classNames)
# %% Model fitting and prediction
n_alphas = 50
reg_strength = np.logspace(-1, 4, n_alphas)

predictions = np.empty(shape=(1, 5))
x_scores = np.empty(shape=(0, folds))

for c in reg_strength:
    # Multinomial logistic regression
    logreg = lm.LogisticRegression(solver='lbfgs',
                                   multi_class='multinomial',
                                   tol=1e-4,
                                   intercept_scaling=True,
                                   random_state=1,
                                   C=c)
    x_val_score = cross_val_score(logreg, X_train, y_train, cv=folds)
    x_scores = np.vstack((x_scores, x_val_score))
    # p = cross_val_predict(logreg, X_test, y_test, cv=folds)
    # predictions.append(p)
"""
logreg.fit(X_train,y_train)
p = logreg.predict(X_test)
predictions.append(p)
prec = precision_score(y_test, p, average='micro')
rec = recall_score(y_test, p, average='micro')
"""

colors = ['#FF5733', '#6C3483', '#229954', '#4C4C4C', '#0000FF']

mean = np.mean(x_scores, axis=1)
std = np.std(x_scores, axis=1)
plt.errorbar(reg_strength, mean, yerr=std, fmt='o', c='#0000FF')

plt.xscale('log')
plt.show()
# To display coefficients use print(logreg.coef_). For a 4 class problem with a 
# feature space, these weights will have shape (4, 2).
"""
# Number of miss-classifications
print('Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}'.format(np.sum(logreg.predict(X_test)!=y_test),len(y_test)))
print('procentage miss-classifications {0}'.format(np.sum((logreg.predict(X_test)!=y_test)/len(y_test))*100))

predict = lambda x: np.argmax(logreg.predict_proba(x),1)

"""
