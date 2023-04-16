import math
import numpy as np


def CentroidDistance(data_array, centroids):
    distance = np.zeros((len(data_array), len(centroids)))
    label = np.zeros(len(distance))

    for i in range(len(data_array)):
        if (i % (len(data_array)/10)) < 1:
            print(f"Progress: {math.floor(i*100/len(data_array))}%")
        for j in range(len(distance[i])):
            distance[i][j] = np.linalg.norm(data_array[i][:64] - centroids[j])

    return label + np.argmin(distance, axis=1)


def LabelData(data_file, centroid_file, file_name):
    print("Labelling:", data_file)
    with open(centroid_file, 'rb') as opened_file:
        centroids = np.load(opened_file)

    if isinstance(data_file, str):
        with open(data_file, 'rb') as opened_file:
            data = np.load(opened_file)
    else:
        data = data_file

    label = CentroidDistance(data, centroids)
    label_data = data[:len(label)]
    label_data = np.concatenate((label_data, np.c_[label]), axis=1)

    np.save(file_name, label_data)


    return label_data


#Call function label data and pass it an unlablled data file, centroid file, and new file name to create
#a npy file with centroid labelled data

# if __name__ == "__main__":
#
#
#     LabelData("nonletter_data_array.npy",
#               "200centroids/centroid_data_200centroids.npy",
#               "200centroids/nonletter_data_200centroids")
#     LabelData("unshuffled_letter_data.npy",
#               "200centroids/centroid_data_200centroids.npy",
#               "200centroids/letter_data_200centroids")

