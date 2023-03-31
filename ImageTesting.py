from PIL import Image
import numpy
import time
import random
import HeatMapBuilder
import KTester as kt
import image_processor_8x8 as ip8
import MultiOutputMLP as mlp
import LabelToLetterTranslation as l2l
import get_word_images as gwi
import os

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

    offset, image_data, image_width, image_height, img_pixel_data = ip8.single_image_processor(
        image_path="images_to_process/nutrition-label.jpg", save_file=False, crop_image=False, resize_by_height=False, remove_bad_samples=False)

    image_array = kt.LabelData(image_data,
                 centroid_file_path,
                 f"{centroid_folder}/Calories_data_150centroids_alex")

    X = mlp.load_testing_data(image_array, centroid_file_path)

    heatMap = HeatMapBuilder.HeatMap(image_width, image_height, len(X), offset, img_pixel_data)
    heatMap.print_dimensions()

    #Test with MLP
    bo_predictions = mlp.run_model(X, bo_hidden_weights, bo_output_weights)
    hits = numpy.nonzero(bo_predictions)
    for hit in hits[0]:
        heatMap.update_heat_map(hit)

    heatMap.print_heat_map(superimpose=False)

    gwi.get_word_images(nutrition_label_image="images_to_process/nutrition-label.jpg", heatmap_image="heatmap.jpg")

    load_word_images = gwi.get_word_image_paths("words")
    counter = 0
    letter_recognitions_text_file = open("LetterRecognitions.txt", "w")
    binary_filtered_text_file = open("LetterRecognitionsBinaryFiltered.txt", "w")
    translated_text_file = open("LetterRecognitionsTranslatedFiltered.txt", "w")
    predicted_word_text_file = open("PredictedWords.txt", "w")

    for file in load_word_images:

        offset, image_data, image_width, image_height, img_pixel_data = ip8.single_image_processor(
            image_path=f"{file}", save_file=False, crop_image=True,
            resize_by_height=True, remove_bad_samples=True)

        if len(image_data) > 0:

            image_array = kt.LabelData(image_data,
                                       centroid_file_path,
                                       f"{centroid_folder}/{counter}Calories_data_150centroids_alex")

            X = mlp.load_testing_data(image_array, centroid_file_path)

            bo_predictions = mlp.run_model(X, bo_hidden_weights, bo_output_weights)
            mo_predictions = mlp.run_model(X, mo_hidden_weights, mo_output_weights)
            hits = numpy.nonzero(bo_predictions)

            letter_recognitions_text_file.write(f"{file}: {str(mo_predictions)}")
            letter_recognitions_text_file.write("\n")
            binary_filtered_text_file.write(f"{file}: {str(bo_predictions)}")
            binary_filtered_text_file.write("\n")
            empty_array = numpy.zeros(len(bo_predictions))
            empty_array[hits] = mo_predictions[hits]
            if len(empty_array[hits]) > 0:
                letter_translation_array = numpy.vectorize(l2l.labels_as_letters.get)(empty_array[hits])
                translated_text_file.write(f"{file}: {str(letter_translation_array)}")
                translated_text_file.write("\n")

                word = l2l.translate_letters_to_words(letter_translation_array)
                predicted_word_text_file.write(f"{file}: {word}")
                predicted_word_text_file.write("\n")
                print(word)

            counter += 1

    letter_recognitions_text_file.close()
    binary_filtered_text_file.close()
    translated_text_file.close()
    predicted_word_text_file.close()

    end = time.time()

    print("Testing Time:", end - start)

    time_string = "Testing Time: " + str(end - start)

    # open text file
    text_file = open("TestingTime.txt", "w")
    text_file.write(time_string)
    text_file.close()

    exit()