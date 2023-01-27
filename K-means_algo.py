import numpy as np
import matplotlib.pyplot
import random
import time
import math

sample_size = 1000

def NormalizeData(data_array):

    data_array = data_array.astype(float)

    for i in range(sample_size):
        mean = np.mean(data_array[i][:64])
        sd = np.std(data_array[i][:64])
        data_array[i][:64] -= mean
        if sd > 0:
            data_array[i][:64] /= sd

    return data_array


def GenerateCentroids():

    lower_bound = 0
    upper_bound = 255
    centroids = np.random.randint(lower_bound, upper_bound, size=(64, 64))

    return centroids


def CentroidDistance(data_array, centroids):

    #distance = np.zeros((len(data_array), len(centroids)))
    distance = np.zeros((sample_size, len(centroids)))

    # Every point should have a single distance value to every centroid
    # Every point therefore has 64 distances attached to it


    for i in range(sample_size):
        for j in range(len(distance[i])):
            for q in range(len(centroids[j])):
                distance[i][j] += (centroids[j][q] - data_array[i][q])**2
            distance[i][j] = math.sqrt(distance[i][j])

    #Each row in the distance var contains 64 elements representing the distance
    # from the corresponding data_array row to each centroid

    return distance


def LabelData(distance):

    label = np.zeros(len(distance))

    for i in range(len(distance)):
        label[i] = np.argmin(distance[i])

    return label


def CentroidMean(label, data_array, centroids):

    new_centroids = np.zeros((64, 64))

    for i in range(len(label)):
        new_centroids[int(label[i])] += data_array[i][:64]

    unique, counts = np.unique(label, return_counts=True)
    label_count = dict(zip(unique, counts))

    for key in label_count:
        new_centroids[int(key)] = new_centroids[int(key)]/label_count[key]

    for x in range(len(new_centroids)):
        if not np.any(new_centroids[x]):
            new_centroids[x] = centroids[x]

    return new_centroids

def CentroidDifference(new_centroid,centroids):

    difference = np.zeros(64)

    for i in range(len(centroids)):
        for j in range(len(centroids[i])):
            difference[i] += (new_centroid[i][j]-centroids[i][j])**2
        difference[i] = math.sqrt(difference[i])

    return difference


def FeatureMapping(x,c,label):

    out = np.zeros(sample_size)
    for i in range(sample_size):
        z = np.linalg.norm(x[i][:64]-c[int(label[i])])
        mean = np.mean(x[i][:64])
        norm = mean - z
        out[i] = max(0.0, (norm))

    return out


if __name__ == "__main__":

    #read in data from numpy file
    with open('Unshuffled_datafile.npy', 'rb') as opened_file:
        data_array = np.load(opened_file)

    #shuffle data
    indexArray = list(range(np.size(data_array, 0)))
    random.shuffle(indexArray)
    data_array = data_array[indexArray]
    data_array = data_array[:12500]

    #Normalize
    #
    # data_array = NormalizeData(data_array)


    #Kmeans
    centroids = GenerateCentroids()
    previous_centroids = centroids
    difference = np.zeros(64)
    difference.fill(200)

    while np.max(difference) > 1:

        distance = CentroidDistance(data_array, centroids)
        label = LabelData(distance)
        new_centroids = CentroidMean(label, data_array, centroids)
        difference = CentroidDifference(new_centroids, centroids)
        centroids = new_centroids
        print(difference)
        print(np.max(difference))

    #Feature Mapping
    feature_map = FeatureMapping(data_array, centroids, label)

    #squish data 0-1 or 0-10





