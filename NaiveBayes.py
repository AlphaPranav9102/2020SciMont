import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

linnerud = datasets.load_linnerud()

X_train, X_test, y_train, y_test = train_test_split(linnerud.data, linnerud.target, test_size=0.3,random_state=109)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print((X_test.shape[0]-(y_test != y_pred).sum())/X_test.shape[0])   