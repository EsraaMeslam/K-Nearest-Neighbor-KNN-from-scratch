import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

cmap = ListedColormap(['r','b','g'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.scatter(X[:,2],X[:,3], c=y ,cmap=cmap, edgecolor='k')


model=KNN(k=5)
model.fit(X_train,y_train)
prediction=model.predict(X_test)

print(prediction) 

accurancy= np.sum(prediction==y_test)/len(y_test)
print(accurancy)


