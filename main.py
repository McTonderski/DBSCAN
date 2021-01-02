# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False

    def visit(self):
        self.visited = True


class DBSCAN:
    def __init__(self, dataPD, minNegh, epsilon):
        self.data = dataPD
        self.minN = minNegh
        self.eps = epsilon
        self.dist = []
        self.classify()

    def classify(self):
        print(self.data.shape[0])
        for i in range(self.data.shape[0]):
            # distance = pow(pow(self.data['f1'][i], 2) + pow(self.data['f2'][i], 2), 1/2)
            # angle = self.data['f2'][i] / self.data['f1'][i]
            self.dist.append(Point(self.data['f1'][i], self.data['f2'][i]))

        self.findNeigbours()

    def findNeigbours(self):
        resN = 0
        self.neigh = dict()
        self.temp = []
        resP = 0
        for j in range(1000 - 1):
            for i in range(j + 1, 1000):
                diff = pow(pow(self.dist[i].x - self.dist[j].x, 2) + pow(self.dist[i].y - self.dist[j].y, 2), 1 / 2)
                if diff <= self.eps:
                    # self.found()
                    resP += 1
                    self.temp.append(self.dist[j])
                else:
                    resN += 1
            self.neigh[j] = self.temp.copy()

        for i in self.neigh:
            print(i)
        print(resP)
        print(resN)


    def found(self):
        print('found')


X, y_true = make_blobs(n_samples=1000, centers=3, cluster_std=0.99, random_state=3042019)
df = pd.DataFrame(X, columns = ['f1', 'f2'])
DBSCAN(df, 2, 0.11)
sns.lmplot(data=df, x='f1', y='f2', fit_reg=False, scatter_kws={"color": "#eb6c6a"}).set(title = 'Wykres punktowy zbioru')
plt.show()

