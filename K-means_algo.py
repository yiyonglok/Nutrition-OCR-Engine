import numpy
import matplotlib.pyplot
import random
import time

if __name__ == "__main__":

    #read in data from numpy file
    with open('Unshuffled_datafile.npy', 'rb') as opened_file:
        data_array = numpy.load(opened_file)

    #shuffle data
    indexArray = list(range(numpy.size(data_array, 0)))
    random.shuffle(indexArray)
    data_array = data_array[indexArray]

    data_array = data_array[:12500]

