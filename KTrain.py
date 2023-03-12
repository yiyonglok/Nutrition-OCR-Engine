import numpy as np
import matplotlib.pyplot
import random
import multiprocessing
import time
import math


def NormalizeData(data_array):

    data_array = data_array.astype(float)

    for i in range(SAMPLE_SIZE):
        mean = np.mean(data_array[i][:64])
        sd = np.std(data_array[i][:64])
        data_array[i][:64] -= mean
        if sd > 0:
            data_array[i][:64] /= sd

    return data_array

def SquishData(data_array):

    return (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))


def GenerateCentroids():

    lower_bound = 0
    upper_bound = 255
    centroids = np.random.randint(lower_bound, upper_bound, size=(CENTROID_COUNT , 64))

    return centroids

def CentroidDistance(data_array, centroids):

    distance = np.zeros((SAMPLE_SIZE, len(centroids)))

    for i in range(SAMPLE_SIZE):
        for j in range(len(distance[i])):
            distance[i][j] = np.linalg.norm(data_array[i][:64]-centroids[j])

    return distance

def LabelData(distance):

    label = np.zeros(len(distance))

    return (label + np.argmin(distance, axis=1))


def CentroidMean(label, data_array, centroids):

    new_centroids = np.zeros((CENTROID_COUNT, 64))

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

    difference = np.zeros(CENTROID_COUNT)

    for i in range(len(centroids)):
        difference[i] = np.linalg.norm(new_centroid[i]-centroids[i])

    return difference

def FeatureMapping(x,c,label):

    out = np.zeros(SAMPLE_SIZE)
    for i in range(SAMPLE_SIZE):
        z = np.linalg.norm(x[i][:64]-c[int(label[i])])
        mean = np.mean(x[i][:64])
        norm = mean - z
        out[i] = max(0.0, (norm))

    return out

def calculate_z(letter_array_i, neuron_weights):
    dot_product = -np.dot(letter_array_i, neuron_weights)
    z_value = 1 / (1 + np.exp(dot_product))
    return z_value



if __name__ == "__main__":

    #read in data from numpy file
    with open('Unshuffled_datafile.npy', 'rb') as opened_file:
        data_array = np.load(opened_file)

    #shuffle data
    indexArray = list(range(np.size(data_array, 0)))
    data_array = data_array[indexArray]

    SAMPLE_SIZE = len(data_array)
    CENTROID_COUNT = 150

    #Squish Data
    # data_array = SquishData(data_array)

    #Normalize
    # data_array = NormalizeData(data_array)

    #Kmeans
    centroids = GenerateCentroids()
    previous_centroids = centroids
    difference = np.zeros(CENTROID_COUNT)
    difference.fill(200)

    while np.max(difference) > 50:

        distance = CentroidDistance(data_array, centroids)
        label = LabelData(distance)
        new_centroids = CentroidMean(label, data_array, centroids)
        difference = CentroidDifference(new_centroids, centroids)
        centroids = new_centroids
        print(difference)
        print(np.max(difference))

    #Feature Mapping
    feature_map = FeatureMapping(data_array, centroids, label)

    #Centroid Labelling
    new_array = data_array[:len(label)]
    new_array = np.concatenate((new_array, np.c_[label]), axis=1)


    np.save("centroid_data_150points", centroids)
    np.save("data_array_150points", new_array)







