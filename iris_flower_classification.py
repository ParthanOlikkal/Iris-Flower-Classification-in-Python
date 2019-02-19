# -*- coding: utf-8 -*-
"""
Spyder Editor

Iris Flower Classification (Supervised Learning)
"""


#Check the versions of library

#Python Version
import sys
print('Python: {}'.format(sys.version))

#scipy Version
import scipy
print('scipy: {}'.format(scipy.__version__))

#numpy Version
import numpy
print('numpy: {}'.format(numpy.__version__))

#matplotlib Version
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

#pandas Version
import pandas
print ('pandas: {}'.format(pandas.__version__))

#sklearn Version
import sklearn
print("sklearn: {}".format(sklearn.__version__))


#Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#Load Dataset
url = "https://raw.githubusercontent.com/ParthanOlikkal/Iris-Flower-Classification/master/Iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width','class']
dataset = pandas.read_csv(url, names = names)

#Shape
print(dataset.shape)

#head
print(dataset.head(20))

#descriptions
print(dataset.describe())

#Class distribution
print(dataset.groupby('class').size())

#Univariate Plots
#box and whisker plots
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show()

#Histograms
dataset.hist()
plt.show()

#Multivariate Plots
#scatter plot matrix
scatter_matrix(dataset)
plt.show()

#Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,test_size = validation_size, random_state = seed)


#Test options and evalutation metric
seed = 7
scoring = 'accuracy'

#Spot check Algorithms
models = []
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma = 'auto')))

#Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
    print(msg)

#Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Make predictions on validation dataset
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation,predictions))

#SVM
svm = SVC();
svm.fit(X_train, Y_train)
pred = svm.predict(X_validation)
print(accuracy_score(Y_validation, pred))
print(confusion_matrix(Y_validation,pred))
print(classification_report(Y_validation, pred))