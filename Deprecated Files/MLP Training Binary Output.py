import numpy
import matplotlib.pyplot
import multiprocessing
import random
import warnings
import time
import centroid_dictionary_builder as cdb
import KTester as kt


alpha = 0.001


def calculate_z(data_array_i, neuron_weights):
    dot_product = -numpy.dot(data_array_i, neuron_weights)
    z_value = 1 / (1 + numpy.exp(dot_product))
    return z_value


def calculate_neuron_weights(data_array_i, neuron_weights, error):
    neuron_weights = neuron_weights + alpha * error * data_array_i
    return neuron_weights


warnings.filterwarnings('ignore')
#numpy.set_printoptions(suppress=True)

if __name__ == "__main__":

    with open('../100centroids/data_array_100centroids.npy', 'rb') as opened_file:
        letter_array = numpy.load(opened_file)

    with open('../100centroids/nonletter_labeled_data_100centroids.npy', 'rb') as opened_file:
        non_letter_array = numpy.load(opened_file)


    letter_centroid_data = letter_array[:,-1] #get centroid labels for datasets
    non_letter_centroid_data = non_letter_array[:,-1]

    #half the letter data by truncating half of each letter's data set
    #letter_centroid_data = numpy.reshape(letter_centroid_data, (62, int(len(letter_centroid_data) / 62)))
    #letter_centroid_data = letter_centroid_data[:,:-int((len(letter_centroid_data[0])/2))]
    #letter_centroid_data = numpy.ndarray.flatten(letter_centroid_data)

    #put 8x8 samples back into 32x32 images
    letter_centroid_data = numpy.reshape(letter_centroid_data,(int(len(letter_centroid_data)/16), 16))
    non_letter_centroid_data = numpy.reshape(non_letter_centroid_data, (int(len(non_letter_centroid_data) / 16), 16))


    # put letter/nonletter centroid data through translator
    letter_data = cdb.build_translated_letter_centroid_labels(letter_centroid_data)
    non_letter_data = cdb.build_translated_letter_centroid_labels(non_letter_centroid_data)

    data_array = numpy.concatenate((letter_data, non_letter_data), axis=0)

    Bias = numpy.full((len(data_array),1), 1)

    Class_one = numpy.full((len(letter_data), 1), 1)
    Class_two = numpy.full((len(non_letter_data), 1), 0)
    Classes = numpy.concatenate((Class_one, Class_two), axis=0)

    #concatenate Bias and Classes to a super letter_array
    data_array = numpy.concatenate((Bias, data_array), axis=1)
    data_array = numpy.concatenate((data_array, Classes), axis=1)

    #shuffle data array for training
    indexArray = list(range(numpy.size(data_array, 0)))
    random.shuffle(indexArray)
    data_array = data_array[indexArray]

    #letter_array = letter_array[:25000]

    #Code to count many letters/non-letters were randomly selected
    count_non = 0
    count_letter = 0

    for i in data_array:
        if i[-1] == 0:
            count_non += 1
        else:
            count_letter += 1

    print("Non-letters:", count_non)
    print("Letters:", count_letter)


    # Splitting up image and classifier data into their own arrays
    data_classifiers = data_array.T[-1]
    data_array = data_array[:, :-1]

    # define number of hidden layers
    NEURONS = 50 + 1  # Extra 1 weight for bias
    ATTRIBUTE_COUNT = len(data_array[0])  # bias already added

    # weights pre-initialized
    neuron_weights = numpy.random.rand(NEURONS * (ATTRIBUTE_COUNT))
    neuron_weights = neuron_weights * 2 - 1
    neuron_weights = neuron_weights.reshape((NEURONS, ATTRIBUTE_COUNT))
    output_weights = numpy.random.rand(NEURONS)
    output_weights = output_weights * 2 - 1

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    print("\nLogical Threads being used:", multiprocessing.cpu_count() - 1)

    # MLP Training
    EPOCHS = 100

    previous_error = 0
    same_error_count = 0
    total_error_list = []

    start = time.time()  # Start Timer

    for epoch in range(EPOCHS):
        total_error = 0
        print("Progress:", str(epoch * 100 / EPOCHS) + "%")
        for i in range(len(data_array)):

            # forward calculation
            hidden_layer_z = numpy.ones(NEURONS)

            # zip and send data to function for z values
            data_array_i = numpy.full((NEURONS, len(data_array[i, :])), data_array[i])
            values = list(zip(data_array_i, neuron_weights))
            return_value = pool.starmap(calculate_z, values)

            if return_value:
                hidden_layer_z = numpy.array(return_value)
            return_value.clear()
            values.clear()

            dot_product_output = -numpy.dot(hidden_layer_z, output_weights)
            output_z = 1 / (1 + numpy.exp(dot_product_output))

            # prediction
            prediction = round(output_z, 0)

            total_error = total_error + abs(data_classifiers[i] - prediction)

            # error propagation
            output_error = output_z * (1 - output_z) * (data_classifiers[i] - output_z)
            error = hidden_layer_z * (1 - hidden_layer_z) * (output_error * output_weights)

            # update weights
            values = list(zip(data_array_i, neuron_weights, error))
            return_value = pool.starmap(calculate_neuron_weights, values)

            if return_value:
                neuron_weights = numpy.array(return_value)
            return_value.clear()
            values.clear()

            output_weights = output_weights + alpha * output_error * hidden_layer_z

        total_error_list.append(total_error)

        if previous_error != total_error_list[-1]:
            previous_error = total_error_list[-1]
            same_error_count = 0
        else:
            same_error_count += 1

        if same_error_count > 3:
            print("Training Completed.")
            break

    end = time.time()

    numpy.save("neuron_weights", neuron_weights)
    numpy.save("output_weights", output_weights)

    print("Number of errors during training:", total_error_list[-1])
    print("Percent of errors during training:", total_error_list[-1] / len(data_array) * 100, "%")

    matplotlib.pyplot.plot(total_error_list)
    matplotlib.pyplot.title(label=f"Alpha: {alpha}")
    matplotlib.pyplot.show()

    #with open('neuron_weights.npy', 'rb') as opened_file:
    #    neuron_weights = numpy.load(opened_file)

    #with open('output_weights.npy', 'rb') as opened_file:
    #    output_weights = numpy.load(opened_file)

    #print("neuron_weights:", neuron_weights)
    #print("output_weights:", output_weights)
    print("Training Time:", end - start)

    time_string = "Training Time: " + str(end - start)

    # open text file
    text_file = open("../TrainingTime.txt", "w")
    text_file.write(time_string)
    text_file.close()
