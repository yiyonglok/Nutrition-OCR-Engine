from PIL import Image
import numpy
import time
import random
import multiprocessing
import centroid_dictionary_builder as cdb
import CentroidPrinter as cp
import HeatMapBuilder


def calculate_z(letter_array_i, neuron_weights):
    dot_product = -numpy.dot(letter_array_i, neuron_weights)
    z_value = 1 / (1 + numpy.exp(dot_product))
    return z_value


def save_letter(letter, name):
    image = Image.fromarray(letter)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(name + ".jpg")


def rebuild_32x32(letter_data, sample_index):
    letter_data = numpy.reshape(letter_data, (16, 64))
    print(len(letter_data[0]))

    temp_array = []

    for i in range(len(letter_data)):
        temp_array.append(numpy.reshape(letter_data[i], (8, 8)))

    #print(len(temp_array[0]))
    temp_image = numpy.block([[temp_array[0], temp_array[1], temp_array[2], temp_array[3]],
                              [temp_array[4], temp_array[5], temp_array[6], temp_array[7]],
                              [temp_array[8], temp_array[9], temp_array[10], temp_array[11]],
                              [temp_array[12], temp_array[13], temp_array[14], temp_array[15]]])

    #temp_image = numpy.reshape(letter_data[0], (8, 8))
    print(temp_image)
    save_letter(temp_image, "image" + str(sample_index))


def rebuild_32x32_again(letter_data, sample_index):
    print(len(letter_data[0]))

    temp_array = []

    for i in range(len(letter_data)):
        temp_array.append(numpy.reshape(letter_data[i], (8, 8)))

    #print(len(temp_array[0]))
    temp_image = numpy.block([[temp_array[0], temp_array[1], temp_array[2], temp_array[3]],
                              [temp_array[4], temp_array[5], temp_array[6], temp_array[7]],
                              [temp_array[8], temp_array[9], temp_array[10], temp_array[11]],
                              [temp_array[12], temp_array[13], temp_array[14], temp_array[15]]])

    #temp_image = numpy.reshape(letter_data[0], (8, 8))
    print(temp_image)
    save_letter(temp_image, "nutrition" + str(sample_index))


if __name__ == "__main__":

    with open('neuron_weights.npy', 'rb') as opened_file:
        neuron_weights = numpy.load(opened_file)

    with open('output_weights.npy', 'rb') as opened_file:
        output_weights = numpy.load(opened_file)

    with open('image_data_with_centroids.npy', 'rb') as opened_file:
        letter_array = numpy.load(opened_file)

    with open('100centroids/centroid_data_100centroids.npy', 'rb') as opened_file:
        centroid_data = numpy.load(opened_file)

    print(len(letter_array[0]))

    #sample_index = 6167
    #index_calculator = sample_index * 16

    #temp_matrix = letter_array[index_calculator:index_calculator+16]
    #temp_matrix = temp_matrix [:,:-1]
    #print("len_temp_matrix[0]:", len(temp_matrix[0]))
    #rebuild_32x32_again(temp_matrix, sample_index)

    letter_centroid_data = letter_array[:,-1] #get centroid labels for datasets
    letter_centroid_data = numpy.reshape(letter_centroid_data,(int(len(letter_centroid_data)/16), 16))


    # put letter/nonletter centroid data through translator
    letter_data = cdb.build_translated_letter_centroid_labels(letter_centroid_data)


    #rebuild_32x32(letter_data[sample_index], sample_index)


    Bias = numpy.full((len(letter_data),1), 1)

    #concatenate Bias and Classes to a super letter_array
    letter_array = numpy.concatenate((Bias, letter_data), axis=1)


    NEURONS = 50+1

    detected_letters = 0
    detected_letters_index = []
    not_a_letter = 0

    print("Rows of data:", len(letter_array))

    # confusion matrix
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    print("\nLogical Threads being used:", multiprocessing.cpu_count() - 1)

    start = time.time()#Start Timer

    print(letter_array[0])

    heatMap = HeatMapBuilder.HeatMap(756, 1008, len(letter_array), 4)
    heatMap.print_dimensions()

    for i in range(len(letter_array)):
        print("Sample: ", i)
        # forward calculation
        hidden_layer_z = numpy.ones(NEURONS)

        # zip and send data to function for z values
        letter_array_i = numpy.full((NEURONS, len(letter_array[i, :])), letter_array[i])
        values = list(zip(letter_array_i, neuron_weights))
        return_value = pool.starmap(calculate_z, values)

        if return_value:
            hidden_layer_z = numpy.array(return_value)
        return_value.clear()
        values.clear()

        dot_product_output = -numpy.dot(hidden_layer_z, output_weights)
        output_z = 1 / (1 + numpy.exp(dot_product_output))

        # prediction
        prediction = round(output_z, 0)
        if prediction > 0:
            detected_letters += 1
            detected_letters_index.append(i)

        # Confusion Matrix Logic
        if prediction == 1:
            heatMap.update_heat_map(i)
        else:
            not_a_letter += 1

        print("prediction", prediction)

        #if i > 10000:
        #    break

    end = time.time()

    print("Hits:", detected_letters)
    print("letter %:", detected_letters*100 / len(letter_array), "%")
    print("Non_letter hits:", not_a_letter)
    print("Non_letter %:", not_a_letter*100 / len(letter_array), "%")

    print("Testing Time:", end - start)

    time_string = "Testing Time: " + str(end - start)

    #open text file
    text_file = open("TestingTime.txt", "w")
    text_file.write(time_string)
    text_file.close()

    heatMap.print_heat_map()

    exit()
