import numpy as np
import string

def centroid_dictionary_builder():
    #init dictionary with alphanumeric chars
    chars = string.digits + string.ascii_uppercase + string.ascii_lowercase
    char_centroids_dict = {char:None for char in chars}

    #load centroid data
    with open('centroid_data.npy', 'rb') as opened_file:
        centroids = np.load(opened_file)

    #flatten centroid data into single dimension
    centroids = centroids.flatten()

    #1016 images/char * 16 centroids/image = 16256 centroids/char
    prev_index = 0
    curr_index = 16256

    #populate dictionary with each char's centroids
    for char in chars:
        char_centroids_dict[char] = centroids[prev_index:curr_index]
        prev_index = curr_index
        curr_index += 16256
    
    return char_centroids_dict
