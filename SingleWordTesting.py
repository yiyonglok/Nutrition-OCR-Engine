from PIL import Image
import numpy
import time
import random
import HeatMapBuilder
import KTester as kt
import image_processor_8x8 as ip8
import MultiOutputMLP as mlp
import LabelToLetterTranslation as l2l


if __name__ == "__main__":
    centroid_folder = '150centroids_alex_shrunk_lowercase'
    bo_hidden_weights_path = f'{centroid_folder}/bo_hidden_weights.npy'
    bo_output_weights_path = f'{centroid_folder}/bo_output_weights.npy'
    mo_hidden_weights_path = f'{centroid_folder}/mo_hidden_weights.npy'
    mo_output_weights_path = f'{centroid_folder}/mo_output_weights.npy'
    centroid_file_path = f'{centroid_folder}/centroid_data_150centroids_alex_shrunk_lowercase.npy'

    with open(bo_hidden_weights_path, 'rb') as opened_file:
        bo_hidden_weights = numpy.load(opened_file)
    with open(bo_output_weights_path, 'rb') as opened_file:
        bo_output_weights = numpy.load(opened_file)
    with open(mo_hidden_weights_path, 'rb') as opened_file:
        mo_hidden_weights = numpy.load(opened_file)
    with open(mo_output_weights_path, 'rb') as opened_file:
        mo_output_weights = numpy.load(opened_file)
    with open(centroid_file_path, 'rb') as opened_file:
        centroid_data = numpy.load(opened_file)

    start = time.time()  # Start Timer

    offset, image_data, image_width, image_height, img_pixel_data, max_height_index, max_width_index = ip8.single_image_processor(image_path="words/word_2_0.jpg",
                                                                               save_file=True, crop_image=True, resize_by_height=True, remove_bad_samples=True)


    image_array = kt.LabelData(image_data,
                 centroid_file_path,
                 f"{centroid_folder}/Calories_data_150centroids_alex")

    X = mlp.load_testing_data(image_array, centroid_file_path)

    #Test with MLP
    bo_predictions = mlp.run_model(X, bo_hidden_weights, bo_output_weights)
    mo_predictions = mlp.run_model(X, mo_hidden_weights, mo_output_weights)

    #update heatmap based on predictions
    hits = numpy.nonzero(bo_predictions)

    # open text file
    text_file = open("LetterRecognitions.txt", "w")
    text_file.write(str(mo_predictions))
    text_file.close()

    text_file = open("LetterRecognitionsBinaryFiltered.txt", "w")
    text_file.write(str(mo_predictions[hits]))
    text_file.close()

    text_file = open("LetterRecognitionsTranslatedFiltered.txt", "w")
    empty_array = numpy.zeros(len(bo_predictions))
    empty_array[hits] = mo_predictions[hits]
    letter_translation_array = numpy.vectorize(l2l.labels_as_letters.get)(empty_array[hits])
    text_file.write(str(letter_translation_array))
    text_file.close()

    cleaned_mo_predictions, bo_predictions = l2l.remove_stacked_samples(mo_predictions, bo_predictions, max_height_index, max_width_index)
    hits = numpy.nonzero(bo_predictions)

    # after cleanup
    empty_array = numpy.zeros(len(bo_predictions))
    empty_array[hits] = cleaned_mo_predictions[hits]
    letter_translation_array = numpy.vectorize(l2l.labels_as_letters.get)(empty_array[hits])

    word = l2l.translate_letters_to_words(letter_translation_array)
    print(word)
    numpy.save("word_prediction", word)

    end = time.time()

    print("Testing Time:", end - start)

    time_string = "Testing Time: " + str(end - start)

    #open text file
    text_file = open("TestingTime.txt", "w")
    text_file.write(time_string)
    text_file.close()

    exit()