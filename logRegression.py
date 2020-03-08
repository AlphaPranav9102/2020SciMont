from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn import datasets

linnerud = datasets.load_linnerud()

X_train, X_test, y_train, y_test = train_test_split(linnerud.data, linnerud.target, test_size=0.3,random_state=109)

clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))