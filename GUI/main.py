import ImageProcessing
import numpy as np
import ImageProcessor
import ImageTesting
import KTrain
import KTester
import MOMLPTraining
import DataReduce
import random
import UI
import DataPrinter
import os
import image_processor_8x8 as ip8
import get_word_images as gwi

if __name__ == "__main__":

    UI.Test().run()
    # offset, image_data, image_width, image_height, img_pixel_data, x, y = ip8.single_image_processor(image_path="images_to_process/nutrition-label.jpg", save_file=False, crop_image=False,remove_bad_samples=False, resize_by_height=False)
    # with open('PredictedWords.npy', 'rb') as opened_file:
    #     predictions = np.load(opened_file)
    # Data = np.genfromtxt("PredictedWords.txt", dtype=str,
    #                      encoding=None, delimiter=":")
    # print(Data[0,1])
    # DataPrinter.image_resize("UIassets/NoImage.png")
    # with open('bo_hidden_weights.npy', 'rb') as opened_file:
    #     bo_hidden_weights = np.load(opened_file)
    # with open('bo_output_weights.npy', 'rb') as opened_file:
    #     bo_output_weights = np.load(opened_file)
    # with open('mo_hidden_weights.npy', 'rb') as opened_file:
    #     mo_hidden_weights = np.load(opened_file)
    # with open('mo_output_weights.npy', 'rb') as opened_file:
    #     mo_output_weights = np.load(opened_file)
    # with open('centroid_data_150centroids_alex_shrunk_lowercase.npy', 'rb') as opened_file:
    #     centroid_data = np.load(opened_file)
    #
    # ImageTesting.RunTest(bo_hidden_weights,bo_output_weights,mo_hidden_weights,mo_output_weights,centroid_data)
    # gwi.get_word_images(nutrition_label_image="label.jpg", heatmap_image="heatmap.jpg")


    # DataPrinter.printData("Char74k_32x32_cleanedv2")

    ## STEP 1 - IMPORT AND LABEL LETTER DATA & RESIZE AND IMPORT NON-LETTER DATA

    # unshuffled_letter_data_v3 = ImageProcessor.read_letter_images("Char74k_32x32_cleanedv3")
    # np.save("unshuffled_letter_data_cleanedv3", unshuffled_letter_data_v3)

    #resize non-letter images to 32x32

    # for path in os.listdir("non_letters"):
    #     # check if current path is a file
    #     if os.path.isfile(os.path.join("non_letters", path)):
    #         ImageProcessor.image_resize("non_letters/"+path)

    # non_letter_data_newest = ImageProcessor.read_nonletter_images("non_letters")
    # np.save("no_letter_data_newest", non_letter_data_newest)

    #non letter is now spliced and labelled
    # with open('no_letter_data_newest.npy', 'rb') as opened_file:
    #     nonletter_data_array = np.load(opened_file)
    #
    # DataPrinter.nonLetterCollage("non_letters")





    ## STEP 2 - IMPORT DATA / LABEL NON-LETTER DATA

    # with open('nonletter_data_array.npy', 'rb') as opened_file:
    #     nonletter_data_array = np.load(opened_file)
    #
    # with open('unshuffled_letter_data_cleaned_faizan.npy', 'rb') as opened_file:
    #     letter_data_array = np.load(opened_file)
    #
    # zero_array = np.zeros((len(nonletter_data_array), 1))
    # nonletter_data_array = np.concatenate((nonletter_data_array, zero_array), axis=1)
    #
    # np.save(f"nonletter_data_array_class_labels", nonletter_data_array

    # ## STEP 3 - RUN KTRAIN ON LABELLED LETTER DATA

    # with open('unshuffled_letter_data_cleanedv3.npy', 'rb') as opened_file:
    #     letter_data_array = np.load(opened_file)
    #
    # KTrain.RunKTrain(letter_data_array)

    #SAVE KTRAIN CENTROID IMAGES

    # DataPrinter.SaveCentroidImages("150Centroids/centroid_data_150centroids.npy")
    # DataPrinter.centroidCollage("150Centroids/centroid_data_150centroids.npy",150)
    # DataPrinter.image_resize("UIassets/CentroidImages/collage_150.jpg")


    ## STEP 4 - RUN KTEST TO LABEL NON-LETTER DATA WITH CENTROIDS FROM (3)

    # with open('nonletter_data_cleanedv3.npy', 'rb') as opened_file:
    #     nonletter_data_array = np.load(opened_file)

    # KTester.LabelData('nonletter_data_cleanedv3.npy', 'centroid_data_150centroids_cleanedv3.npy', 'nonletter_data_array_150centroids_cleanedv3.npy')

    ## STEP 5 - TRAIN MLP WITH DATA GENERATED IN STEPS 1-4

    # with open('150Centroids/nonletter_data_array_150centroids.npy', 'rb') as opened_file:
    #     nonletter_data_array = np.load(opened_file)
    #
    # with open('150Centroids/data_array_150centroids.npy', 'rb') as opened_file:
    #     letter_data_array = np.load(opened_file)
    #
    # print(len(nonletter_data_array[0]))
    # print(len(letter_data_array[0]))

    # MOMLPTraining.TrainMPL()



    ## EXTRA (1) - REDUCE DATASET CODE

    # load_letters = DataReduce.CollectData("Char74k_32x32_cleanedv3/Sample033")
    # ref_values = DataReduce.CollectRefNumber(load_letters)
    #
    # file_names = DataReduce.CollectData("Char74k_32x32_cleanedv3")
    # DataReduce.RemoveFiles(file_names,ref_values)

    # with open('nonletter_data_array_class_labels.npy', 'rb') as opened_file:
    #     nonletter_data_array = np.load(opened_file)
    #
    # with open('unshuffled_letter_data_cleanedv3.npy', 'rb') as opened_file:
    #     letter_data_array = np.load(opened_file)
    #
    # print(len(letter_data_array))
    # print(len(nonletter_data_array))
    # np.random.shuffle(nonletter_data_array)
    # nonletter_data_array = nonletter_data_array[0:len(letter_data_array)]
    #
    # np.save("nonletter_data_cleanedv3", nonletter_data_array)


    ##




























