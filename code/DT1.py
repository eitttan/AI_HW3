from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

# load data
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

trainUrl = 'https://grades.cs.technion.ac.il/grades.cgi?cabgfhafd47798fa3870e2db53fc73+2+236501+Winter2019-2020+hw' \
           '/WCFiles/train.csv+7960 '
trainSet = pd.read_csv('C:/Users/eittt/Documents/semester11/bina/hw3/train.csv')
testSet = pd.read_csv('C:/Users/eittt/Documents/semester11/bina/hw3/test.csv')
testData = testSet.iloc[:, 0:-1]
testLabel = testSet.iloc[:, -1]
# TODO : relative loading

 # fitted model
X = trainSet.iloc[:, 0:-1]
Y = trainSet.iloc[:, -1]
for minLeaf in [1, 3, 9, 27]:
    clf = tree.DecisionTreeClassifier(min_samples_leaf = minLeaf)
    clf = clf.fit(X, Y)
    testPredicted = clf.predict(testData)
    print(minLeaf)
    print(confusion_matrix(testLabel, testPredicted))

graph = tree.plot_tree(clf)
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())