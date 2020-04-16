import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import recall_score, precision_score, confusion_matrix, classification_report
from auxiliary import one_out_of_k, calc_distribution, trim_ticks
from imblearn.under_sampling import RandomUnderSampler


def update_weights(dic, lower=0.2, upper=3, step=0.2):
    if dic[1] >= upper:
        return dic

    dic[5] += step
    if dic[5] >= upper:
        for i in list(range(5, 1, -1)):
            dic[i] = lower
            dic[i-1] += step
    return dic

file_path = "./res/spotify-data-apr-2019.csv"
df_data = pd.read_csv(file_path)  # data as pandas DataFrame format
attribute_names = df_data.columns.values  # numpy array of attribute names
df_data.sample(frac=1)  # shuffle rows of data
data = df_data.values  # all data in numpy array format

vf = 0.2  # fraction of the data reserved for validation (usually between 0.2 and 0.3)
#N = data.shape[0]  # number of rows in the data set
#M = data.shape[1]  # number of columns in the data set
y = data[:, 13]  # class belonging to each row in normal format
X = np.delete(data, 13, axis=1)
X = preprocessing.scale(X)
folds = 10  # fold for k-folds x-validation

rus = RandomUnderSampler(random_state=0)
X, y = rus.fit_resample(X, y)
N = X.shape[0]  # number of rows in the data set
M = X.shape[1]  # number of columns in the data set

# create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vf)

weight_dict = dict(zip([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]))  # pre-defined weights for each class
criterion = 'gini'
min_sample_tests = 20
tests = list(range(min_sample_tests))
recall = []
precision = []
tests = 4000
length = 0

for i in range(tests):
    dec_tree = tree.DecisionTreeClassifier(criterion=criterion,
                                           class_weight=weight_dict,
                                           min_samples_split=5)  # create model
    dec_tree.fit(X_train, y_train)
    y_est = dec_tree.predict(X_test)
    recall.append(recall_score(y_test, y_est, average='macro'))
    precision.append(precision_score(y_test, y_est, average='macro'))
    weight_dict = update_weights(weight_dict)
    if weight_dict[1] >= 3:
        length = i + 1
        break

x = list(range(length))
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
labels = ['recall', 'precision']
colors = ['#0000FF', '#4C4C4C']
p = [recall, precision]

for i in range(2):
    axs[i].plot(x, p[i], c=colors[i])
    axs[i].set_ylabel(labels[i])
    axs[i].grid()

print("Maximum calculated recall %0.5f" % max(recall))
print("Recall calculated at index %i" % recall.index(max(recall)))
weight_dict = dict(zip([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]))
for i in range(recall.index(max(recall))):
    update_weights(weight_dict)

print("Optimal weights for recall:")
for k, v in weight_dict.items():
    print("Class: {}, Weight: {}".format(k, v))


print("Maximum calculated precision %0.5f" % max(precision))
print("Precision calculated at index %i" % precision.index(max(precision)))
weight_dict = dict(zip([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]))
for i in range(precision.index(max(precision))):
    update_weights(weight_dict)

print("Optimal weights for Precision:")
for k, v in weight_dict.items():
    print("Class: {}, Weight: {}".format(k, v))

plt.show()

dec_tree = tree.DecisionTreeClassifier(criterion=criterion,
                                       class_weight=dict(zip([1, 2, 3, 4, 5], [2, 0.2, 0.2, 0.2, 2])) ,
                                       min_samples_split=5)  # create model
dec_tree.fit(X_train, y_train)

scores_b = cross_val_score(dec_tree, X_train, y_train, cv=folds)  # calculates x-validation scores
print("X-validation used: %i-folds" % folds)
print("X-validation accuracy: %0.5f (+/- %0.5f)" % (scores_b.mean(), scores_b.std() * 2))

predictions = cross_val_predict(dec_tree, X_test, y_test, cv=folds)  # calculates the test accuracy
classification_report = classification_report(y_test,
                                                      predictions)  # calculates the R^2 value between true- & predicted values
print(classification_report)

_, p_count = np.unique(predictions, return_counts=True)
classes, t_count = np.unique(y_test, return_counts=True)

barWidth = 0.25
r1 = [x - barWidth / 2 for x in np.arange(5) + 1]
r2 = [x + barWidth / 2 for x in np.arange(5) + 1]

plt.bar(r1, t_count, width=barWidth, label='true', color='#0000FF')
plt.bar(r2, p_count, width=barWidth, label='predicted', color='#4C4C4C')
plt.xlabel("Classes")
plt.ylabel("Quantity")
plt.legend()
plt.grid()
plt.show()

# create confusion matrix
CM = confusion_matrix(y_test,
                              predictions,
                              [1, 2, 3, 4, 5])

plt.matshow(CM, cmap=plt.cm.winter)
plt.colorbar()
for i in range(len(CM)):
    for j in range(len(CM)):
        text = plt.text(j, i, CM[i, j],
                        ha="center", va="center", color="#000000")
plt.xticks(np.arange(5), ['1', '2', '3', '4', '5'])
plt.yticks(np.arange(5), ['1', '2', '3', '4', '5'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

dec_tree = tree.DecisionTreeClassifier(criterion=criterion,
                                       class_weight=dict(zip([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])) ,
                                       min_samples_split=5)  # create model
dec_tree.fit(X_train, y_train)

print('procentage miss-classifications {0}'.format(np.sum((dec_tree.predict(X_test)!=y_test)/len(y_test))*100))

dec_tree = tree.DecisionTreeClassifier(criterion=criterion,
                                       class_weight=dict(zip([1, 2, 3, 4, 5], [2, 0.2, 0.2, 0.2, 2])) ,
                                       min_samples_split=5)  # create model
dec_tree.fit(X_train, y_train)

print('procentage miss-classifications {0}'.format(np.sum((dec_tree.predict(X_test)!=y_test)/len(y_test))*100))