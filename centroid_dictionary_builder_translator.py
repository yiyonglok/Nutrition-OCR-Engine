import numpy as np

def build_centroid_dictionary():

    #load centroid data
    with open('centroid_data.npy', 'rb') as file:
        centroid_data = np.load(file)

    #init and build centroid dictionary, keys/centroids are integers 0-63
    centroid_dictionary = {centroid:None for centroid in range(len(centroid_data))}
    for centroid in centroid_dictionary:
        centroid_dictionary[centroid] = centroid_data[centroid]

    return centroid_dictionary

def build_translated_letter_centroid_labels():

    #load letter centroids
    with open('letter_centroid_labels.npy', 'rb') as file:
        letter_centroid_labels = np.load(file)

    #load centroid data
    with open('centroid_data.npy', 'rb') as file:
        centroid_data = np.load(file)

    #build dictionary to translate centroids
    centroid_dictionary = build_centroid_dictionary()

    #init translated/expanded letter centroid labels array, 62992x1024 (16*64)
    translated_letter_centroid_labels = np.empty((len(letter_centroid_labels), len(letter_centroid_labels[0])*len(centroid_data[0])))

    #translate each row and assign to translated array
    for image_row, letter_image in enumerate(letter_centroid_labels):
        translated_centroids = np.array([])
        for centroid in letter_image:
            translated_centroids = np.append(translated_centroids, centroid_dictionary[centroid])
        translated_letter_centroid_labels[image_row] = translated_centroids
    
    return translated_letter_centroid_labels

if __name__ == "__main__":

    centroid_dictionary = build_centroid_dictionary()
    #print(centroid_dictionary)
    translated_letter_centroid_labels = build_translated_letter_centroid_labels()
    #print(translated_letter_centroid_labels.shape)
    np.save("translated_letter_centroid_labels", translated_letter_centroid_labels)
