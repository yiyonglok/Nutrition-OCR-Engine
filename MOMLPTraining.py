import matplotlib.pyplot
import numpy
import warnings
import time
import MultiOutputMLP as mlp
import collections


if __name__ == "__main__":
    alpha = 0.0001
    hidden_neurons = 1000
    EPOCHS = 200
    binary_output = False
    directory_path = "50centroids"
    letter_data_path = "labelled_letterdata.npy"
    non_letter_data_path = "labelled_nonletterdata.npy"
    centroid_file_path = f"{directory_path}/centroid_data_50points.npy"

    X = mlp.load_training_data(letter_data_path, non_letter_data_path, centroid_file_path, binary_output)
    X = mlp.shuffle_data(X)
    #X = mlp.shuffle_data(mlp.generate_random_data(10000))
    #print(len(X))
    #print(X[0])

    X_classifiers, X = mlp.separate_label(X)
    #print("X_classifier collection:", collections.Counter(X_classifiers))

    output_neurons = len(set(X_classifiers))


    hidden_weights, output_weights, X_classifiers_vectors = mlp.initialize_neural_network(hidden_neurons,
                                                                                      output_neurons,
                                                                                      X,
                                                                                      X_classifiers)

    # MLP Training
    start = time.time()  # Start Timer
    total_error_list, hidden_weights, output_weights = mlp.train_model(EPOCHS,
                                       X, X_classifiers, X_classifiers_vectors,
                                       alpha, hidden_weights, output_weights)

    mlp.save_weights(directory_path, output_neurons, hidden_weights, output_weights)

    end = time.time()
    print("Training Time:", end - start)
    time_string = "Training Time: " + str(end - start)

    matplotlib.pyplot.plot(total_error_list)
    matplotlib.pyplot.title(label=f"Alpha: {alpha}, Epochs: {EPOCHS}")
    matplotlib.pyplot.show()