import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import datasets

linnerud = datasets.load_linnerud()

X_train, X_test, y_train, y_test = train_test_split(linnerud.data, linnerud.target, test_size=0.3,random_state=109)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


