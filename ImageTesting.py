from PIL import Image
import numpy
import time
import random
import multiprocessing
import centroid_dictionary_builder as cdb
import CentroidPrinter as cp
import HeatMapBuilder
import KTester as kt


def calculate_z(data_array_i, neuron_weights):
    dot_product = -numpy.dot(data_array_i, neuron_weights)
    z_value = 1 / (1 + numpy.exp(dot_product))
    return z_value


def save_letter(letter, name):
    image = Image.fromarray(letter)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    file_path = name + ".jpg"
    image.save(file_path)


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
    save_letter(temp_image, "ImageTestingOuput/image" + str(sample_index))


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
    save_letter(temp_image, "ImageTestingOuput/nutrition" + str(sample_index))


if __name__ == "__main__":

    with open('100c_001a_new_neuron_weights.npy', 'rb') as opened_file:
        neuron_weights = numpy.load(opened_file)

    with open('100c_001a_new_output_weights.npy', 'rb') as opened_file:
        output_weights = numpy.load(opened_file)

    with open('100centroids/centroid_data_100centroids.npy', 'rb') as opened_file:
        centroid_data = numpy.load(opened_file)

    start = time.time()  # Start Timer

    image_array = kt.LabelData("nl_2_8x8_data_single.npy",
                 "100centroids/centroid_data_100centroids.npy",
                 "100centroids/nonletter_labeled_data_100centroids.npy")

    letter_centroid_data = image_array[:,-1] #get centroid labels for datasets
    letter_centroid_data = numpy.reshape(letter_centroid_data,(int(len(letter_centroid_data)/16), 16))

    # put letter/nonletter centroid data through translator
    data_array = cdb.build_translated_letter_centroid_labels(letter_centroid_data)

    #sample_index = 6167
    #rebuild_32x32(data_array[sample_index], sample_index)


    Bias = numpy.full((len(data_array),1), 1)

    #concatenate Bias and Classes to a super letter_array
    data_array = numpy.concatenate((Bias, data_array), axis=1)

    NEURONS = 50+1

    detected_letters = 0
    detected_letters_index = []
    not_a_letter = 0

    print("Rows of data:", len(data_array))

    # confusion matrix
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    print("\nLogical Threads being used:", multiprocessing.cpu_count() - 1)

    print(data_array[0])

    heatMap = HeatMapBuilder.HeatMap(756, 1008, len(data_array), 4)
    heatMap.print_dimensions()

    for i in range(len(data_array)):
        print("Sample: ", i)
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
        if prediction > 0:
            detected_letters += 1
            detected_letters_index.append(i)

        # Confusion Matrix Logic
        if prediction == 1:
            None
        else:
            not_a_letter += 1
            heatMap.update_heat_map(i)

        print("prediction", prediction)

        #if i > 10000:
        #    break

    end = time.time()

    print("Hits:", detected_letters)
    print("letter %:", detected_letters*100 / len(data_array), "%")
    print("Non_letter hits:", not_a_letter)
    print("Non_letter %:", not_a_letter*100 / len(data_array), "%")

    print("Testing Time:", end - start)

    time_string = "Testing Time: " + str(end - start)

    #open text file
    text_file = open("TestingTime.txt", "w")
    text_file.write(time_string)
    text_file.close()

    heatMap.print_heat_map()

    exit()
