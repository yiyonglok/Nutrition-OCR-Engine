
import numpy
import matplotlib.pyplot
import random
import warnings
import time
import collections
import centroid_dictionary_builder as cdb
import data_reshaper


warnings.filterwarnings('ignore')

def activation_function(data_matrix, neuron_weights):
    #print("neuron w:", neuron_weights)
    #print("X", data_matrix)
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

def generate_random_data(class_data_size):
    mean_one = [-6, 0, -6]
    covariance_one = [[2, 1, 1], [1, 3, 1], [1, 1, 2]]
    mean_two = [6, 0, 6]
    covariance_two = [[3, 1, 1], [1, 2, 1], [1, 1, 2]]
    mean_three = [10, 2, 10]
    covariance_three = [[3, 1, 1], [1, 3, 1], [1, 1, 1]]

    x_one = numpy.random.multivariate_normal(mean_one, covariance_one, class_data_size)
    x_two = numpy.random.multivariate_normal(mean_two, covariance_two, class_data_size)
    x_three = numpy.random.multivariate_normal(mean_three, covariance_three, class_data_size)

    X = numpy.concatenate((x_one, x_two, x_three), axis=0)
    X = numpy.concatenate((numpy.ones((3 * class_data_size, 1)), X), axis=1)

    classifier_one = 0
    classifier_two = 1
    classifier_three = 2

    X_classifiers = numpy.full((class_data_size, 1), classifier_one)
    X_classifiers = numpy.concatenate((X_classifiers, numpy.full((class_data_size, 1), classifier_two)), axis=0)
    X_classifiers = numpy.concatenate((X_classifiers, numpy.full((class_data_size, 1), classifier_three)), axis=0)

    X = numpy.concatenate((X, X_classifiers), axis=1)
    return X

def load_data(letter_datapath, non_letter_datapath):
    print("Loading data...")
    with open(letter_datapath, 'rb') as opened_file:
        letter_array = numpy.load(opened_file)

    with open(non_letter_datapath, 'rb') as opened_file:
        non_letter_array = numpy.load(opened_file)

    print("Preparing training data...")
    letter_centroid_data = letter_array[:, -1]  # get centroid labels for datasets
    non_letter_centroid_data = non_letter_array[:, -1]

    # reshape labelled array into collections of the original images represented by labels in 4x4 labels
    letter_centroid_data = numpy.reshape(letter_centroid_data, (int(len(letter_centroid_data) / 16), 16))
    non_letter_centroid_data = numpy.reshape(non_letter_centroid_data, (int(len(non_letter_centroid_data) / 16), 16))

    # put letter/nonletter centroid data through translator
    letter_data = cdb.build_translated_letter_centroid_labels(letter_centroid_data)
    non_letter_data = cdb.build_translated_letter_centroid_labels(non_letter_centroid_data)

    X = numpy.concatenate((letter_data, non_letter_data), axis=0)

    #reshape the 32x32's properly to recreate the letter representations
    X = data_reshaper.reshape_letter_data(X, save_images=False, save_file=False)

    Bias = numpy.full((len(X), 1), 1)

    Classes = []
    for i in range(0, len(letter_array), 16):
        Classes.append(letter_array[i][-2])

    Classes = numpy.reshape(numpy.array(Classes), (len(Classes), 1))
    Class_not = numpy.full((len(non_letter_data), 1), 0)
    Classes = numpy.concatenate((Classes, Class_not), axis=0)

    # concatenate Bias and Classes to a super letter_array
    X = numpy.concatenate((Bias, X), axis=1)
    X = numpy.concatenate((X, Classes), axis=1)
    return X


def shuffle_data(X):
    # shuffle data
    index_array = list(range(numpy.size(X, 0)))
    random.shuffle(index_array)
    X = X[index_array]
    return X


def separate_label(X):
    # split data from classifier labels after shuffling
    X_classifiers = X.T[-1]
    X = X[:, :-1]
    return X_classifiers, X


def initialize_neural_network(HNEURONS, ONEURONS, X, X_classifiers):
    # Initialize Neural Network
    ATTRIBUTE_COUNT = numpy.size(X, axis=1)
    HNEURONS = HNEURONS + 1  # +1 for bias for input to the output neurons

    hidden_weights = initialize_neuron_weights(HNEURONS, ATTRIBUTE_COUNT)
    output_weights = initialize_neuron_weights(ONEURONS, HNEURONS)
    one_hot_encoded_classifiers = one_hot_encode(X_classifiers, ONEURONS)
    return hidden_weights, output_weights, one_hot_encoded_classifiers


def initialize_neuron_weights(neurons, input_size):
    neuron_weights = numpy.random.rand(neurons, input_size)
    neuron_weights = neuron_weights * 2 - 1
    return neuron_weights


def one_hot_encode(X_classifiers, ONEURONS):
    # One hot encode the entire classifier label list
    X_classifiers = [int(index) for index in X_classifiers]
    one_hot_encoded_classifiers = numpy.zeros((len(X_classifiers), ONEURONS))
    one_hot_encoded_classifiers[numpy.arange(len(X_classifiers)), X_classifiers] = 1
    #print(one_hot_encoded_classifiers)
    #print(len(one_hot_encoded_classifiers))
    #print(len(one_hot_encoded_classifiers[0]))
    return one_hot_encoded_classifiers


def forward_propagation(X, hidden_weights, output_weights):
    hidden_layer_z = activation_function(X, hidden_weights)
    # hidden_layer_z = numpy.dot(X, neuron_weights.T)
    # print("Hidden Layer Z:", hidden_layer_z)
    # get output z vector
    output_z = numpy.dot(hidden_layer_z, output_weights.T)
    # print("Output Z:", output_z)
    softmax_z = softmax(output_z)
    # print("Softmax Z:", softmax_z)
    return hidden_layer_z, softmax_z

def backwards_propagation(X, X_classifiers_vectors, alpha, hidden_weights, output_weights, hidden_layer_z, softmax_z):
    output_error = (softmax_z - X_classifiers_vectors).T
    # print("Output Error:", output_error)
    delta_output_weights = (1 / len(X) * numpy.dot(output_error, hidden_layer_z))
    # print(delta_output_weights)
    error = (numpy.dot(output_weights.T, output_error)).T * (relu_derivative(hidden_layer_z))
    delta_hidden_weights = (1 / len(X) * numpy.dot(X.T, error)).T
    # print(delta_hidden_weights)
    hidden_weights = hidden_weights - alpha * delta_hidden_weights
    output_weights = output_weights - alpha * delta_output_weights
    return hidden_weights, output_weights

def predict(softmax_z):
    prediction = numpy.argmax(softmax_z, axis=1)
    return prediction

def train_model(EPOCHS, X, X_classifiers, X_classifiers_vectors, alpha, hidden_weights, output_weights):
    total_error_list = []
    for epoch in range(EPOCHS):
        total_error = 0
        if (epoch % (EPOCHS/10)) == 0:
            print("Progress:", str(epoch * 100 / EPOCHS) + "%")

        hidden_layer_z, softmax_z = forward_propagation(X, hidden_weights, output_weights)

        hidden_weights, output_weights = backwards_propagation(X, X_classifiers_vectors,
                                                               alpha,
                                                               hidden_weights, output_weights,
                                                               hidden_layer_z, softmax_z)

        prediction = predict(softmax_z)
        total_error = numpy.sum(prediction != X_classifiers)

        if (epoch % (EPOCHS/10)) == 0:
            #print("predictions: ", prediction)
            #print("expecteds:", X_classifiers)
            print("Total Errors:", total_error)
            print("Accuracy:", (len(X) - total_error) * 100 / len(X), "% \n")

        total_error_list.append(total_error)
    return total_error_list


def save_weights(path, output_neurons, hidden_weights, output_weights):
    if output_neurons > 1:
        hw_file_path = f"{path}/mo_hidden_weights"
        ow_file_path = f"{path}/mo_output_weights"
        numpy.save(hw_file_path, hidden_weights)
        numpy.save(ow_file_path, output_weights)
    else:
        hw_file_path = f"{path}/bo_hidden_weights"
        ow_file_path = f"{path}/bo_output_weights"
        numpy.save(hw_file_path, hidden_weights)
        numpy.save(ow_file_path, output_weights)
    print("Weights saved")


if __name__ == "__main__":

    X = shuffle_data(generate_random_data(1000))
    X_classifiers, X = separate_label(X)

    alpha = 0.001
    hidden_neurons = 5
    output_neurons = 3
    EPOCHS = 1000

    hidden_weights, output_weights, X_classifiers_vectors = initialize_neural_network(hidden_neurons,
                                                                                      output_neurons,
                                                                                      numpy.size(X, axis=1),
                                                                                      X_classifiers)

    # MLP Training
    start = time.time()  # Start Timer
    total_error_list = train_model(EPOCHS, X, X_classifiers_vectors, alpha, hidden_weights, output_weights)
    end = time.time()
    print("Training Time:", end - start)
    time_string = "Training Time: " + str(end - start)

    matplotlib.pyplot.plot(total_error_list)
    matplotlib.pyplot.title(label=f"Alpha: {alpha}")
    matplotlib.pyplot.show()
