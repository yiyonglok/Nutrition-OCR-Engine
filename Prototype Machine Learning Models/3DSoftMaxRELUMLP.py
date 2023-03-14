
import numpy
import matplotlib.pyplot
import random
import warnings
import time

alpha = 0.001


def activation_function(data_matrix, neuron_weights):
    z_value = relu(numpy.dot(data_matrix, neuron_weights.T))
    return z_value

def update_weights(data_array_i, neuron_weights, error):
    neuron_weights = neuron_weights + alpha * error * data_array_i
    return neuron_weights

def relu(x):
    return numpy.maximum(x, 0)

def relu_derivative(z):
    return z > 0

def softmax(z):
  '''Return the softmax output of a vector.'''
  z = z.T
  softmax_z = numpy.round( numpy.exp(z) / sum(numpy.exp(z)), 3)
  return softmax_z.T


if __name__ == "__main__":

    mean_one = [-6, 0, -6]
    covariance_one = [[2, 1, 1], [1, 3, 1], [1, 1, 2]]

    mean_two = [6, 0, 6]
    covariance_two = [[3, 1, 1], [1, 2, 1], [1, 1, 2]]

    mean_three = [10, 2, 10]
    covariance_three = [[3, 1, 1], [1, 3, 1], [1, 1, 1]]

    class_data_size = 4

    x_one = numpy.random.multivariate_normal(mean_one, covariance_one, class_data_size)
    x_two = numpy.random.multivariate_normal(mean_two, covariance_two, class_data_size)
    x_three = numpy.random.multivariate_normal(mean_three, covariance_three, class_data_size)

    X = numpy.concatenate((x_one, x_two, x_three), axis=0)
    X = numpy.concatenate((numpy.ones((3*class_data_size, 1)), X), axis=1)

    classifier_one = 0
    classifier_two = 1
    classifier_three = 2

    X_classifiers = numpy.full((class_data_size, 1), classifier_one)
    X_classifiers = numpy.concatenate((X_classifiers, numpy.full((class_data_size, 1), classifier_two)), axis=0)
    X_classifiers = numpy.concatenate((X_classifiers, numpy.full((class_data_size, 1), classifier_three)), axis=0)

    X = numpy.concatenate((X, X_classifiers), axis=1)

    #shuffle data
    index_array = list(range(numpy.size(X, 0)))
    random.shuffle(index_array)
    X = X[index_array]

    #split data from classifier labels after shuffling
    X_classifiers = X.T[-1]
    X = X[:, :-1]

    # Initialize Neural Network
    HNEURONS = 5 + 1  #+1 for bias for input to the output neurons
    ONEURONS = 3 #No extra bias, output neurons must match number of softmax classes
    ATTRIBUTE_COUNT = len(X[0])  # bias already added to the input values

    neuron_weights = numpy.random.rand(HNEURONS, ATTRIBUTE_COUNT)
    neuron_weights = neuron_weights * 2 - 1
    output_weights = numpy.random.rand(ONEURONS, HNEURONS)
    output_weights = output_weights * 2 - 1

    #One hot encode the entire classifier label list
    X_classifiers = [int(index) for index in X_classifiers]
    X_classifiers_vectors = numpy.zeros((len(X_classifiers), ONEURONS))
    X_classifiers_vectors[numpy.arange(len(X_classifiers)), X_classifiers] = 1
    print(X_classifiers_vectors)

    # MLP Training
    EPOCHS = 1
    previous_error = 0
    same_error_count = 0
    total_error_list = []

    start = time.time()  # Start Timer

    for epoch in range(EPOCHS):
        total_error = 0
        if (epoch % (EPOCHS/10)) == 0:
            print("Progress:", str(epoch * 100 / EPOCHS) + "%")

        #print(X)
        #print(neuron_weights)

        hidden_layer_z = activation_function(X, neuron_weights)
        # hidden_layer_z = numpy.dot(X, neuron_weights.T)
        #print(hidden_layer_z)

        #get output z vector
        output_z = numpy.dot(hidden_layer_z, output_weights.T)
        #print("Output Z:", output_z)

        softmax_z = softmax(output_z)
        print("Softmax Z:", softmax_z)

        output_error = (softmax_z - X_classifiers_vectors).T
        print("Output Error:", output_error)
        delta_output_weights = (1 / len(X) * numpy.dot(output_error, hidden_layer_z))
        #print(delta_output_weights)

        error = (numpy.dot(output_weights.T, output_error)).T * (relu_derivative(hidden_layer_z))
        delta_neuron_weights = (1 / len(X) * numpy.dot(X.T, error)).T

        #print(delta_neuron_weights)

        neuron_weights = neuron_weights - alpha * delta_neuron_weights
        output_weights = output_weights - alpha * delta_output_weights

        prediction = numpy.argmax(softmax_z, axis=1)
        total_error = numpy.sum(prediction != X_classifiers)

        if (epoch % (EPOCHS/10)) == 0:
            #print("predictions: ", prediction)
            #print("expecteds:", X_classifiers)
            print("Total Errors:", total_error)
            print("Accuracy:", (len(X) - total_error) * 100 / len(X), "% \n")

        total_error_list.append(total_error)

    end = time.time()
    print("Training Time:", end - start)
    time_string = "Training Time: " + str(end - start)

    matplotlib.pyplot.plot(total_error_list)
    matplotlib.pyplot.title(label=f"Alpha: {alpha}")
    matplotlib.pyplot.show()

