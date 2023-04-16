import matplotlib.pyplot
import numpy
import warnings
import time
import MultiOutputMLP as mlp
import collections


def TrainBOMLP():
    alpha = 0.0001
    hidden_neurons = 1000
    EPOCHS = 300
    binary_output = True
    directory_path = "150centroids"
    letter_data_path = f"{directory_path}/letter_data_150centroids_alex.npy"
    non_letter_data_path = f"{directory_path}/nonletter_data_150centroids_alex.npy"
    centroid_file_path = f"{directory_path}/centroid_data_150centroids_alex.npy"

    X = mlp.load_training_data(letter_data_path, non_letter_data_path, centroid_file_path, binary=binary_output)
    X = mlp.shuffle_data(X)
    #X = mlp.shuffle_data(mlp.generate_random_data(10000))
    #print(len(X))
    #print(X[0])

    X_classifiers, X = mlp.separate_label(X)
    #print("X_classifier collection:", collections.Counter(X_classifiers))

    output_neurons = len(set(X_classifiers))
    #print(output_neurons)

    hidden_weights, output_weights, X_classifiers_vectors = mlp.initialize_neural_network(hidden_neurons,
                                                                                      output_neurons,
                                                                                      X,
                                                                                      X_classifiers)

    # MLP Training
    start = time.time()  # Start Timer
    total_error_list = mlp.train_model(EPOCHS,
                                       X, X_classifiers, X_classifiers_vectors,
                                       alpha, hidden_weights, output_weights)
    end = time.time()
    print("Training Time:", end - start)
    time_string = "Training Time: " + str(end - start)

    mlp.save_weights(directory_path, output_neurons, hidden_weights, output_weights)

    matplotlib.pyplot.plot(total_error_list)
    matplotlib.pyplot.title(label=f"Alpha: {alpha}, Epochs: {EPOCHS}")
    matplotlib.pyplot.show()