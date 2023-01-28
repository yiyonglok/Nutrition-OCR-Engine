import numpy
import matplotlib.pyplot
import multiprocessing
import random
import warnings
import time

alpha = 0.1


def calculate_z(letter_array_i, neuron_weights):
    dot_product = -numpy.dot(letter_array_i, neuron_weights)
    z_value = 1 / (1 + numpy.exp(dot_product))
    return z_value


def calculate_neuron_weights(letter_array_i, neuron_weights, error):
    neuron_weights = neuron_weights + alpha * error * letter_array_i
    return neuron_weights


warnings.filterwarnings('ignore')
# numpy.set_printoptions(suppress=True)

if __name__ == "__main__":

    # with open('testfile.npy', 'rb') as opened_file:
    #    letter_array = numpy.load(opened_file)

    # scikit-image library grayscale conversion equation: Y = 0.2125 R + 0.7154 G + 0.0721 B

    # adds bias to first field
    letter_array = numpy.random.randint(0, high=255, size=(32 * 32 + 2) * 5).reshape((5, 32 * 32 + 2))

    for i in range(len(letter_array)):
        letter_array[i][-1] = random.randint(0, 1)

    for i in range(len(letter_array)):
        letter_array[i][0] = 1

    indexArray = list(range(numpy.size(letter_array, 0)))
    random.shuffle(indexArray)
    letter_array = letter_array[indexArray]

    # letter_array = letter_array[:12500]

    # Code to count many faces/non-faces were randomly selected
    # count_non = 0
    # count_faces = 0

    # for i in letter_array:
    #    if i[-1] == 0:
    #        count_non += 1
    #    else:
    #        count_faces += 1

    # print("Non-faces:", count_non)
    # print("Faces:", count_faces)

    # letter_array = letter_array[:100]

    # Splitting up pre-saved image and classifier data into their own arrays
    letter_classifiers = letter_array.T[-1]
    letter_array = letter_array[:, :-1]

    # define number of hidden layers
    NEURONS = 50 + 1  # Extra 1 weight for bias
    ATTRIBUTE_COUNT = len(letter_array[0])  # bias already added

    # weights pre-initialized
    neuron_weights = numpy.random.rand(NEURONS * (ATTRIBUTE_COUNT))
    neuron_weights = neuron_weights * 2 - 1
    neuron_weights = neuron_weights.reshape((NEURONS, ATTRIBUTE_COUNT))
    output_weights = numpy.random.rand(NEURONS)
    output_weights = output_weights * 2 - 1

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    print("\nLogical Threads being used:", multiprocessing.cpu_count() - 1)

    # MLP Training
    EPOCHS = 15
    previous_error = 0
    same_error_count = 0
    total_error_list = []

    start = time.time()  # Start Timer

    for epoch in range(EPOCHS):
        total_error = 0
        print("Progress:", str(epoch * 100 / EPOCHS) + "%")
        for i in range(len(letter_array)):

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

            total_error = total_error + abs(letter_classifiers[i] - prediction)

            # error propagation
            output_error = output_z * (1 - output_z) * (letter_classifiers[i] - output_z)
            error = hidden_layer_z * (1 - hidden_layer_z) * (output_error * output_weights)

            # update weights
            values = list(zip(letter_array_i, neuron_weights, error))
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

        # if same_error_count > 3:
        # break

    end = time.time()

    numpy.save("neuron_weights", neuron_weights)
    numpy.save("output_weights", output_weights)

    print("Number of errors during training:", total_error_list[-1])
    print("Percent of errors during training:", total_error_list[-1] / len(letter_array) * 100, "%")

    matplotlib.pyplot.plot(total_error_list)
    matplotlib.pyplot.title(label=f"Alpha: {alpha}")
    # matplotlib.pyplot.show()

    with open('neuron_weights.npy', 'rb') as opened_file:
        neuron_weights = numpy.load(opened_file)

    with open('output_weights.npy', 'rb') as opened_file:
        output_weights = numpy.load(opened_file)

    print("neuron_weights:", neuron_weights)
    print("output_weights:", output_weights)
    print("Training Time:", end - start)

    time_string = "Training Time: " + str(end - start)

    # open text file
    text_file = open("TrainingTime.txt", "w")
    text_file.write(time_string)
    text_file.close()