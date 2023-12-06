import numpy as np

from collections import Counter


def Euclidean_Distance(x1,x2):
    dis=np.sqrt(np.sum((x1-x2)**2))
    return dis




class KNN:
    def __init__(self,k=3):
        self.k=k


    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

        
    
    def predict(self,X):
        predictions=[self.predict_(x) for x in X]



    def predict_(self,x):

        distance=[Euclidean_Distance(x,x_train) for x_train in self.X_train ]
        k_indc=np.argsort(distance)[0 :self.k]
        k_nearest_labels=[self.y_train[i] for i in k_indc  ]


        most_common=Counter(k_nearest_labels).most_common()
        return most_common[0][0]
