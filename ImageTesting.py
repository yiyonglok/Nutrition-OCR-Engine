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

    offset, image_data, image_width, image_height, img_pixel_data, quality_samples = ip8.single_image_processor(image_path="images_to_process/white.png",
                                                                               save_file=True, crop_image=False, resize_by_height=False)
    #print(image_width)
    #print(image_height)
    #print(quality_samples)
    #quality_image_data = image_data[quality_samples]
    #print(numpy.shape(quality_image_data))

    print(numpy.shape(quality_samples))
    image_array = kt.LabelData(image_data,
                 centroid_file_path,
                 f"{centroid_folder}/Calories_data_150centroids_alex")

    #quality_image_array = image_array[quality_samples]


    #image_array = f"{centroid_folder}/nutritionlabel_data_150centroids_faizan.npy"

    X = mlp.load_testing_data(image_array, centroid_file_path)
    #X_cleaned = mlp.load_testing_data(quality_image_array, centroid_file_path)

    heatMap = HeatMapBuilder.HeatMap(image_width, image_height, len(X), offset, img_pixel_data)
    heatMap.print_dimensions()

    #Test with MLP
    bo_predictions = mlp.run_model(X, bo_hidden_weights, bo_output_weights)
    print(numpy.shape(bo_predictions))
    #print(numpy.shape(bo_predictions[quality_samples]))
    mo_predictions = mlp.run_model(X, mo_hidden_weights, mo_output_weights)
    #print(numpy.shape(mo_predictions[quality_samples]))
    print(numpy.shape(mo_predictions))

    #bo_predictions_cleaned = mlp.run_model(X_cleaned, bo_hidden_weights, bo_output_weights)
    #mo_predictions_cleaned = mlp.run_model(X_cleaned, mo_hidden_weights, mo_output_weights)

    #update heatmap based on predictions
    hits = numpy.nonzero(bo_predictions)
    #hits_cleaned = numpy.nonzero(bo_predictions[quality_samples])
    for hit in hits[0]:
        heatMap.update_heat_map(hit)

    # open text file
    text_file = open("LetterRecognitions.txt", "w")
    text_file.write(str(mo_predictions[quality_samples]))
    text_file.close()

    text_file = open("LetterRecognitionsBinaryFiltered.txt", "w")
    #text_file.write(str(mo_predictions[quality_samples][hits_cleaned]))
    text_file.write(str(mo_predictions[hits]))
    text_file.close()

    text_file = open("LetterRecognitionsTranslatedFiltered.txt", "w")
    #empty_array = numpy.zeros(len(bo_predictions[quality_samples]))
    empty_array = numpy.zeros(len(bo_predictions))
    #empty_array[hits_cleaned] = mo_predictions[quality_samples][hits_cleaned]
    #letter_translation_array = numpy.vectorize(l2l.labels_as_letters.get)(empty_array[hits_cleaned])
    empty_array[hits] = mo_predictions[hits]
    letter_translation_array = numpy.vectorize(l2l.labels_as_letters.get)(empty_array[hits])
    text_file.write(str(letter_translation_array))
    text_file.close()

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

    heatMap.print_heat_map(superimpose=True)

    exit()