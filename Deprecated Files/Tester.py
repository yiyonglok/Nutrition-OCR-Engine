from PIL import Image
import numpy
import time
import random
import multiprocessing
import centroid_dictionary_builder as cdb


def calculate_z(letter_array_i, neuron_weights):
    dot_product = -numpy.dot(letter_array_i, neuron_weights)
    z_value = 1 / (1 + numpy.exp(dot_product))
    return z_value


def save_letter(letter, name):
    image = Image.fromarray(letter)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(name + ".jpg")


if __name__ == "__main__":

    with open('neuron_weights.npy', 'rb') as opened_file:
        neuron_weights = numpy.load(opened_file)

    with open('output_weights.npy', 'rb') as opened_file:
        output_weights = numpy.load(opened_file)

    with open('../100centroids/data_array_100centroids.npy', 'rb') as opened_file:
        letter_array = numpy.load(opened_file)

    with open('../100centroids/nonletter_labeled_data_100centroids.npy', 'rb') as opened_file:
        non_letter_array = numpy.load(opened_file)


    letter_centroid_data = letter_array[:,-1] #get centroid labels for datasets
    non_letter_centroid_data = non_letter_array[:,-1]

    letter_centroid_data = numpy.reshape(letter_centroid_data,(int(len(letter_centroid_data)/16), 16))
    non_letter_centroid_data = numpy.reshape(non_letter_centroid_data, (int(len(non_letter_centroid_data) / 16), 16))

    print(non_letter_centroid_data[0])

    print(non_letter_centroid_data[0])

    # put letter/nonletter centroid data through translator
    letter_data = cdb.build_translated_letter_centroid_labels(letter_centroid_data)
    nonletter_data = cdb.build_translated_letter_centroid_labels(non_letter_centroid_data)

    letter_array = numpy.concatenate((letter_data, nonletter_data), axis=0)

    Bias = numpy.full((len(letter_array),1), 1)

    Class_one = numpy.full((len(letter_centroid_data), 1), 1)
    Class_two = numpy.full((len(non_letter_centroid_data), 1), 0)
    Classes = numpy.concatenate((Class_one, Class_two), axis=0)

    #concatenate Bias and Classes to a super letter_array
    letter_array = numpy.concatenate((Bias, letter_array), axis=1)
    letter_array = numpy.concatenate((letter_array, Classes), axis=1)


    indexArray = list(range(numpy.size(letter_array, 0)))
    random.shuffle(indexArray)
    letter_array = letter_array[indexArray]

    #letter_array = letter_array[:10000]

    #Code to count many letters/non-letters were randomly selected
    count_non = 0
    count_letter = 0

    for i in letter_array:
        if i[-1] == 0:
            count_non += 1
        else:
            count_letter += 1

    print("Non-letters:", count_non)
    print("Letters:", count_letter)


    # Splitting up image and classifier data into their own arrays
    test_data_classifiers = letter_array.T[-1]
    letter_array = letter_array[:, :-1]

    NEURONS = 50+1

    detected_letters = 0
    detected_letters_index = []

    print("Rows of data:", len(letter_array))

    # confusion matrix
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    print("\nLogical Threads being used:", multiprocessing.cpu_count() - 1)

    start = time.time()#Start Timer

    for i in range(len(letter_array)):
        print("Progress: ", i*100/len(letter_array),"%")
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

        #Confusion Matrix Logic
        if prediction == 1:
            if test_data_classifiers[i] == 1:
                true_positive = true_positive + 1
            else:
                false_positive = false_positive + 1
        else:
            if test_data_classifiers[i] == 1:
                false_negative = false_negative + 1
            else:
                true_negative = true_negative + 1

        print("prediction", prediction)
        print("class", test_data_classifiers[i])

    end = time.time()

    print("\nConfusion Matrix:")
    print("True Positive: ", true_positive, "|  True Negative: ", true_negative)
    print("False Positive: ", false_positive, "|  False Negative: ", false_negative)

    print("Testing Time:", end - start)

    time_string = "Testing Time: " + str(end - start)

    #open text file
    text_file = open("../TestingTime.txt", "w")
    text_file.write(time_string)
    text_file.close()

    exit()
