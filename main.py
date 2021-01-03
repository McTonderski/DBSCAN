# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.datasets import make_blobs, make_moons, make_circles

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False

    def visit(self):
        self.visited = True

    def is_visited(self):
        return self.visited

    def __str__(self):
        return "X: " + str(self.x) + " Y: " + str(self.y)


class DBSCAN:
    def __init__(self, dataPD, minNegh, epsilon):
        self.data = dataPD
        self.minN = minNegh
        self.eps = epsilon
        self.dist = []
        self.classify()

    def classify(self):
        for i in range(self.data.shape[0]):
            # data serialization to separate points
            self.dist.append(Point(self.data['f1'][i], self.data['f2'][i]))

        self.findNeigbours()

    def findNeigbours(self):
        # debugging variables
        resN = 0
        self.neigh = dict()
        self.temp = []
        resP = 0
        # actual distance calculation and result evaluation
        for j in range(1000):
            self.dist[j].visit()
            for i in range(j + 1, 1000):
                # if self.dist[i].is_visited():
                #     continue
                diff = pow(pow(self.dist[i].x - self.dist[j].x, 2) + pow(self.dist[i].y - self.dist[j].y, 2), 1 / 2)
                if diff <= self.eps:
                    # self.found()

                    resP += 1
                    self.temp.append(deepcopy(i))
                    # self.dist[i].visit()
                else:
                    resN += 1
            self.neigh[deepcopy(j)] = deepcopy(self.temp)
            self.temp = []

        self.plot_results()
        print(resP)

    def plot_results(self):
        self.groups = []

        def return_child(dat):
            print(dat)
            res = []
            for p in dat:
                res.append(p)
                for val in return_child(self.neigh[p]):
                    if val not in res:
                        print("res" + str(res))
                        res.append(val)

            return res

        for group in return_child(self.neigh[0]):
            self.groups.append(group)

        print(self.groups)



    def found(self):
        print('found')


X, y_true = make_blobs(n_samples=1000, centers=3, cluster_std=0.98, random_state=3042019)
df = pd.DataFrame(X, columns = ['f1', 'f2'])
DBSCAN(df, 2, 0.55)
sns.lmplot(data=df, x='f1', y='f2', fit_reg=False, scatter_kws={"color": "#eb6c6a"}).set(title = 'Wykres punktowy zbioru')
plt.show()

