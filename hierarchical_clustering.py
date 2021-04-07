# -*- coding: utf-8 -*-
import numpy as np
from math import *


def manhattan_distance(x, y):
    print(x)
    print(y)
    return sum(abs(a - b) for a, b in zip(x, y))


def intersampledist(s1, s2):
    '''
        To be used in case we have one sample and one cluster . It takes the help of one
        method 'interclusterdist' to compute the distances between elements of a cluster(which are
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
                    print("1")
                    dist.append(interclusterdist(s2[i], s1[j]))
                else:
                    print("2")
                    dist.append(manhattan_distance(s2[i][0], s1[j]))
    else:
        for i in range(m):
            for j in range(n):
                if (len(s1[i]) >= len(s2[j])) and str(type(s1[i][0]) != '<class \'list\'>'):
                    print("3")
                    dist.append(interclusterdist(s1[i], s2[j]))
                else:
                    print("4")
                    print(s1[i][0])
                    print(s2[j])
                    dist.append(manhattan_distance(s1[i][0], s2[j]))
    print(dist)
    return min(dist)


def interclusterdist(cl, sample):
    if sample[0] != '<class \'list\'>':
        print("blabla")
        print(sample)
        sample = [sample]
        print(sample)
        print(cl)
    dist = []
    for i in range(len(cl)):
        for j in range(len(sample)):
            print("sss")
            print("sample", sample[j])
            print("cluster", cl[i])
            dist.append(manhattan_distance(cl, sample[j]))
    print("intercluster")
    print(dist)
    return min(dist)


def distance_calculate(sample1, sample2):
    '''
        Distance calulated between two samples. The two samples can be both samples, both clusters or
        one cluster and one sample. If both of them are samples/clusters, then simple norm is used. In other
        cases, we refer it as an exception case and pass the samples as parameter to some function that
        calculates the necessary distance between cluster and a sample
    '''
    print(len(sample1))
    print(len(sample2))
    dist = []
    for i in range(len(sample1)):
        for j in range(len(sample2)):
            if len(sample1) > 1 or len(sample2) > 1:
                print("tessst")
                print(sample1[i])
                print(sample2[j])
                dist.append(intersampledist(sample1[i], sample2[j]))
            else:
                print("test")
                print(sample1[i])
                print(sample2[j])
                dist.append(manhattan_distance(sample1[i], sample2[j]))
    print(dist)
    return min(dist)


def single_linkage(X):
    cluster_size = len(X)
    distance_matrix = np.zeros((cluster_size, cluster_size))
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[0]):
            if i != j:
                distance_matrix[i, j] = float(distance_calculate(X[i], X[j]))
            else:
                distance_matrix[i, j] = 1000
    return distance_matrix


def agglomerative_clustering():
    # initial data
    X = np.array([[0.0, 0.0],
                  [10.0, 10.0],
                  [21.0, 21.0],
                  [33.0, 33.0],
                  [5.0, 27.0],
                  [28.0, 6.0]
                  ])

    Y = np.array([[0.40, 0.53],
                  [0.22, 0.32],
                  [0.35, 0.32],
                  [0.26, 0.19],
                  [0.08, 0.41],
                  [0.35, 0.30],
                  [0.80, 0.98],
                  [0.28, 0.33]
                  ])

    # initialize values and keys
    keys = [[i] for i in range(X.shape[0])]
    values = [[list(X[i])] for i in range(X.shape[0])]
    cluster_size = len(values)

    # preform merging of clusters
    while cluster_size > 1:
        print("Cluster before Clustering:")
        print(keys)
        print(values)
        # calculate distance matrix
        distance_matrix = single_linkage(values)
        print("Distance Matrix:")
        print(distance_matrix)

        # find two clusters with minimum distance
        minimum_clusters = np.where(distance_matrix == distance_matrix.min())[0]
        print('The two nearest clusters are:', keys[minimum_clusters[0]], ' and ', keys[minimum_clusters[1]])

        # add cluster 1 to cluster 2
        value_to_add = values.pop(minimum_clusters[0])
        values[minimum_clusters[0]].append(value_to_add)
        keys[minimum_clusters[0]].append(keys[minimum_clusters[1]])
        keys[minimum_clusters[0]] = [keys[minimum_clusters[0]]]
        temp = keys.pop(minimum_clusters[1])

        print("Cluster after Clustering:")
        print(keys)
        print("\n")
        cluster_size = len(values)


agglomerative_clustering()

