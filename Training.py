import numpy
import matplotlib.pyplot
import multiprocessing
import random
import warnings
import time

def calculate_z(letter_array_i, neuron_weights):
    z_value = 1
    dot_product = -numpy.dot(letter_array_i, neuron_weights)
    z_value = 1 / (1 + numpy.exp(dot_product))
    return z_value

def calculate_error():
    None

warnings.filterwarnings('ignore')
#numpy.set_printoptions(suppress=True)

if __name__ == "__main__":
    #numpy.set_printoptions(suppress=True)

    #with open('testfile.npy', 'rb') as opened_file:
    #    letter_array = numpy.load(opened_file)

    #scikit-image library grayscale conversion equation: Y = 0.2125 R + 0.7154 G + 0.0721 B
    #Adds bias to first field
    letter_array = numpy.random.randint(0, high=255, size=(32*32+2)*5).reshape((5, 32*32+2))

    for i in range(len(letter_array)):
        letter_array[i][-1] = random.randint(0, 1)

    for i in range(len(letter_array)):
        letter_array[i][0] = 1

    indexArray = list(range(numpy.size(letter_array,0)))
    random.shuffle(indexArray)
    letter_array = letter_array[indexArray]

    #letter_array = letter_array[:12500]

    #Code to count many faces/non-faces were randomly selected
    #count_non = 0
    #count_faces = 0

    #for i in letter_array:
    #    if i[-1] == 0:
    #        count_non += 1
    #    else:
    #        count_faces += 1

    #print("Non-faces:", count_non)
    #print("Faces:", count_faces)

    #letter_array = letter_array[:100]

    #Splitting up pre-saved image and classifier data into their own arrays
    letter_classifiers = letter_array.T[-1]
    letter_array = letter_array[:, :-1]

    print(len(letter_array[0]))

    print(letter_array)
    print(letter_classifiers)


    # define number of hidden layers
    NEURONS = 50 + 1 #Extra 1 weight for bias
    ATTRIBUTE_COUNT = len(letter_array[0]) + 1 #Extra 1 weight for bias

    # weights pre-initialized
    neuron_weights = numpy.random.rand(NEURONS * (ATTRIBUTE_COUNT - 1)) #Minus 1 remove one of the biases
    neuron_weights = neuron_weights * 2 - 1
    neuron_weights = neuron_weights.reshape((NEURONS, ATTRIBUTE_COUNT - 1)) #Minus 1 remove one of the biases
    output_weights = numpy.random.rand(NEURONS)
    output_weights = output_weights * 2 - 1

    #print("Output_weights:", output_weights)
    #print("# of Output_weights:", len(output_weights))
    print("# of neuron_weights:", len(neuron_weights))
    print("neuron_weights[0]:", neuron_weights[0])
    #print("# of neuron_weights[0]:", len(neuron_weights[0]))


    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    print("\nLogical Threads being used:", multiprocessing.cpu_count() - 1)

    #MLP Training
    EPOCHS = 1
    alpha = 0.1
    previous_error = 0
    same_error_count = 0
    total_error_list = []

    start = time.time()#Start Timer

    for epoch in range(EPOCHS):
        total_error = 0
        print("Progress:", str(epoch*100/EPOCHS)+"%")
        for i in range(len(letter_array)):

            # forward calculation
            hidden_layer_z = numpy.ones(NEURONS)

            #zip and send data to function for z values
            letter_array_i = numpy.full((NEURONS, len(letter_array[i, :])), letter_array[i, :])
            values = list(zip(letter_array_i, neuron_weights))
            return_value = pool.starmap(calculate_z, values)

            if return_value:
                hidden_layer_z = numpy.array(return_value)
            values.clear()

            dot_product_output = -numpy.dot(hidden_layer_z, output_weights)
            output_z = 1 / (1 + numpy.exp(dot_product_output))

            print(output_z)

            # prediction
            prediction = round(output_z, 0)
            #print("prediction", prediction)
            #print("class", letter_classifiers[i])

            total_error = total_error + abs(letter_classifiers[i] - prediction)
            #print("total error", total_error)

            # error propagation
            error = numpy.ones(NEURONS)
            #error[NEURONS] = z[NEURONS] * (1 - z[NEURONS]) * (letter_classifiers[i] - z[NEURONS])
            #for j in range(NEURONS):
            #    error[j] = z[j] * (1 - z[j]) * (error[NEURONS] * output_weights[j + 1])

            output_error = output_z * (1 - output_z) * (letter_classifiers[i] - output_z)
            #for j in range(NEURONS):
            #    error[j] = hidden_layer_z[j] * (1 - hidden_layer_z[j]) * (output_error * output_weights[j])
            #    print("error[j] =", hidden_layer_z[j], "*", "(1 -", hidden_layer_z[j], ") * (", output_error, "*", output_weights[j], ") = ", error[j])

            new_error = hidden_layer_z * (1 - hidden_layer_z) * (output_error * output_weights)

            print("error:", error)
            print("new_error:", new_error)

'''   
            output_weights[0] = output_weights[0] + alpha * error[NEURONS]
            for j in range(NEURONS):
                output_weights[j + 1] = output_weights[j + 1] + alpha * error[NEURONS] * z[j]
                neuron_weights[j][0] = neuron_weights[j][0] + alpha * error[j]
                for k in range(ATTRIBUTES):
                    neuron_weights[j][k + 1] = neuron_weights[j][k + 1] + alpha * error[j] * letter_array[i, k + 1]
    
        total_error_list.append(total_error)
    
        if previous_error != total_error_list[-1]:
            previous_error = total_error_list[-1]
            same_error_count = 0
        else:
            same_error_count += 1
    
        if same_error_count > 3:
            break
    
    end = time.time()
    
    numpy.save("neuron_weights", neuron_weights)
    numpy.save("output_weights", output_weights)
    
    print("Number of errors during training:", total_error_list[-1])
    print("Percent of errors during training:", total_error_list[-1]/len(letter_array) * 100, "%")
    
    matplotlib.pyplot.plot(total_error_list)
    matplotlib.pyplot.title(label=f"Alpha: {alpha}")
    #matplotlib.pyplot.show()
    
    with open('neuron_weights.npy', 'rb') as opened_file:
        neuron_weights = numpy.load(opened_file)
    
    with open('output_weights.npy', 'rb') as opened_file:
        output_weights = numpy.load(opened_file)
    
    print("neuron_weights:",neuron_weights)
    print("output_weights:",output_weights)
    print("Training Time:", end - start)
    
    time_string = "Training Time: " + str(end - start)
    
    #open text file
    text_file = open("TrainingTime.txt", "w")
    text_file.write(time_string)
    text_file.close()
'''