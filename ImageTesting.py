from PIL import Image
import numpy
import time
import random
import HeatMapBuilder
import KTester as kt
import image_processor_8x8 as ip8
import MultiOutputMLP as mlp



if __name__ == "__main__":
    centroid_folder = '150centroids'
    bo_hidden_weights_path = f'{centroid_folder}/bo_hidden_weights.npy'
    bo_output_weights_path = f'{centroid_folder}/bo_output_weights.npy'
    mo_hidden_weights_path = f'{centroid_folder}/mo_hidden_weights.npy'
    mo_output_weights_path = f'{centroid_folder}/mo_output_weights.npy'
    centroid_file_path = f'{centroid_folder}/centroid_data_150centroids_faizan.npy'

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

    offset, image_data, image_width, image_height = ip8.single_image_processor(image_path="images_to_process/Calories.png",
                                                                               save_file=True)

    image_array = kt.LabelData(image_data,
                 centroid_file_path,
                 f"{centroid_folder}/Calories_data_150centroids_faizan")

    #image_array = f"{centroid_folder}/nutritionlabel_data_150centroids_faizan.npy"

    X = mlp.load_testing_data(image_array, centroid_file_path)

    heatMap = HeatMapBuilder.HeatMap(image_width, image_height, len(X), offset)
    heatMap.print_dimensions()

    #Test with MLP
    bo_predictions = mlp.run_model(X, bo_hidden_weights, bo_output_weights)
    mo_predictions = mlp.run_model(X, mo_hidden_weights, mo_output_weights)

    #update heatmap based on predictions
    hits = numpy.nonzero(bo_predictions)
    for hit in hits[0]:
        heatMap.update_heat_map(hit)

    # open text file
    text_file = open("LetterRecognitions.txt", "w")
    text_file.write(str(mo_predictions))
    text_file.close()

    text_file = open("LetterRecognitionsBinaryFiltered.txt", "w")
    text_file.write(str(mo_predictions[hits]))
    text_file.close()

    end = time.time()

    print("Testing Time:", end - start)

    time_string = "Testing Time: " + str(end - start)

    #open text file
    text_file = open("TestingTime.txt", "w")
    text_file.write(time_string)
    text_file.close()

    heatMap.print_heat_map()

    exit()
