# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import *


class Distances_Repo(object):
    ''' holds all the distance computations necessary to complete clustering '''
    def __init__(self):
        pass

    def manhattan_dist(self, x, y):
        sum = 0
        # for the case any of the inputs is a scalar value
        if str(type(x)) != '<class \'list\'>':
            sum = abs(x) + abs(y[0]-y[1])
        elif str(type(y)) != '<class \'list\'>':
            sum = abs(x[0]-x[-1]) + abs(y)
        # elif str(type(x[0])) == '<class \'list\'>' or str(type(y[0])) == '<class \'list\'>':
        #     for i in range(len(x)):
        #         for j in range(i + 1, len(x)):
        #             sum += self.manhattan_dist(x[i], x[j]) + self.manhattan_dist(y[i], y[j])

        else: sum = abs(x[0]-x[-1]) + abs(y[0]-y[-1])

        return sum

    def intercluster_dist(self, cluster, sample):
        if sample[0] != '<class \'list\'>':
            sample = [sample]
        dist = []
        for i in range(len(cluster)):
            for j in range(len(sample)):
                dist.append(self.manhattan_dist(cluster[i], sample[j]))
        return min(dist)

    def intersample_dist(self, s1, s2):
        '''
            To be used in case we have one sample and one cluster . It takes the help of one
            method 'intercluster_dist' to compute the distances between elements of a cluster(which are
            samples) and the actual sample given.
        '''
        if str(type(s2[0])) != '<class \'list\'>':
            s2 = [s2]
        if str(type(s1[0])) != '<class \'list\'>':
            s1 = [s1]
        m = len(s1)
        n = len(s2)
        dist = []
        if n >= m:
            for i in range(n):
                for j in range(m):
                    if (len(s2[i]) >= len(s1[j])) and str(type(s2[i][0]) != '<class \'list\'>'):
                        dist.append(self.intercluster_dist(s2[i], s1[j]))
                    else:
                        dist.append(self.manhattan_dist(s2[i][0], s1[j]))
        else:
            for i in range(m):
                for j in range(n):
                    if (len(s1[i]) >= len(s2[j])) and str(type(s1[i][0]) != '<class \'list\'>'):
                        dist.append(self.intercluster_dist(s1[i], s2[j]))
                    else:
                        dist.append(self.manhattan_dist(s1[i][0], s2[j]))
        return min(dist)

    def distance_calculate(self, sample1, sample2):
        '''
            Distance calulated between two samples. The two samples can be both samples, both clusters or
            one cluster and one sample. If both of them are samples/clusters, then simple norm is used. In other
            cases, we refer it as an exception case and pass the samples as parameter to some function that
            calculates the necessary distance between cluster and a sample
        '''
        dist = []
        for i in range(len(sample1)):
            for j in range(len(sample2)):
                try:
                    dist.append(self.manhattan_dist(sample1[i], sample2[j]))
                except:
                    dist.append(self.intersample_dist(sample1[i], sample2[j]))
        return min(dist)

    def single_linkage(self, X):
        cluster_size = len(X)
        distance_matrix = np.zeros((cluster_size, cluster_size))
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[0]):
                if i != j:
                    distance_matrix[i, j] = float(self.distance_calculate(X[i], X[j]))
                else:
                    distance_matrix[i, j] = 1000
        return distance_matrix



class Agglomerative_Clustering(object):

    def __init__(self):
        pass

    def testdata_setup(self):
        # initial data
        X = np.array(
            [0.40, 0.53, 0.22, 0.32, 0.35, 0.32, 0.26, 0.19, 0.08, 0.41, 0.35, 0.30, 0.80, 0.98, 0.28, 0.33, 1.0,
             0.77, 0.5, 0.41, 0.55, 0.55, 0.23, 0.56]).reshape(12, 2)
        print(X)

        # display the initial points
        fig = plt.figure()
        fig.suptitle("Scatter plot of points from sample X")
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X[:, 0], X[:, 1])
        plt.show()
        return X

    def agglomerative_clustering(self, distRepo):
        X = self.testdata_setup()
        # initialize values and keys
        keys = [[i] for i in range(X.shape[0])]
        values = [[list(X[i])] for i in range(X.shape[0])]
        cluster_size = len(values)

        # preform merging of clusters
        while cluster_size > 1:
            print("Sample size prior to clustering: ", cluster_size)
            # calculate distance matrix
            distance_matrix = distRepo.single_linkage(values)
            # find two clusters with minimum distance
            minimum_clusters = np.where(distance_matrix == distance_matrix.min())[0]
            print(minimum_clusters)
            # add cluster 1 to cluster 2
            value_to_add = values.pop(minimum_clusters[1])
            values[minimum_clusters[0]].append(value_to_add)

            keys[minimum_clusters[0]].append(keys[minimum_clusters[1]])
            keys[minimum_clusters[0]] = [keys[minimum_clusters[0]]]
            temp = keys.pop(minimum_clusters[1])
            cluster_size = len(values)
            print("Current Sample: ", keys)
            print("Cluster attained: ", keys[minimum_clusters[0]])
            print("Sample size after clustering step: ", cluster_size, "\n")


if __name__ == "__main__":
    distRepo = Distances_Repo()
    aggloCluster = Agglomerative_Clustering()
    aggloCluster.agglomerative_clustering(distRepo)


