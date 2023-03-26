from PIL import Image, ImageOps
import os
import numpy as np


def testing_image_scale(image_path):
    # validation
    if not image_path:
        print("Image not specified")
        exit()

    # get file name
    file_name = image_path.split(".")[0]

    # process image
    image_data_8x8 = []
    with Image.open(image_path) as img:
        # get image dimensions, grayscale, load pixel data
        img = resize_image(img)
        width, height = img.size
        img = ImageOps.grayscale(img)
        img_pixel_data = img.load()

    # determine max index to take sample within image dimensions
    max_width_index = determine_max_valid_index(SAMPLE_SIZE, offset, width)
    max_height_index = determine_max_valid_index(SAMPLE_SIZE, offset, height)

    # take 32x32 sample in 8x8 pieces for each offset step
    count = 0
    for i in range(0, max_height_index + 1, offset):
        print(f"{file_name} : row {i} of {max_height_index}\r", end="")
        for j in range(0, max_width_index + 1, offset):
            sample_32x32 = []
            # build 32x32 sample
            for y_index in range(i, i + SAMPLE_SIZE):
                temp_array = []
                for x_index in range(j, j + SAMPLE_SIZE):
                    temp_array.append(img_pixel_data[x_index, y_index])
                sample_32x32.append(temp_array)
            print("sample before", np.shape(sample_32x32))
            formatted_32x32 = dynamically_crop_image(sample_32x32)


if __name__ == "__main__":
    centroid_folder = '150centroids'
    centroid_file_path = f'{centroid_folder}/centroid_data_150centroids_alex.npy'

    offset, image_data, image_width, image_height = ip8.single_image_processor(
        image_path="processed_images/saturated.png",
        save_file=True)

    print(np.shape(image_data))

    image_array = kt.LabelData(image_data,
                               centroid_file_path,
                               f"{centroid_folder}/training_saturated_data_150centroids_alex")