from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets

linnerud = datasets.load_linnerud()

X_train, X_test, y_train, y_test = train_test_split(linnerud.data, linnerud.target, test_size=0.3,random_state=109)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10000000)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))