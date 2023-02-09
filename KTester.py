import numpy as np


def CentroidDistance(data_array, centroids):
    distance = np.zeros((len(data_array), len(centroids)))
    label = np.zeros(len(distance))

    for i in range(len(data_array)):
        for j in range(len(distance[i])):
            distance[i][j] = np.linalg.norm(data_array[i][:64] - centroids[j])

    return label + np.argmin(distance, axis=1)


def LabelData(data_file, centroid_file,file_name):
    with open(centroid_file, 'rb') as opened_file:
        centroids = np.load(opened_file)

    with open(data_file, 'rb') as opened_file:
        data = np.load(opened_file)

    label = CentroidDistance(data, centroids)
    label_data = data[:len(label)]
    label_data = np.concatenate((label_data, np.c_[label]), axis=1)

    np.save(file_name, label_data)


    return label_data


#Call function label data and pass it an unlablled data file, centroid file, and new file name to create
#a npy file with centroid labelled data



