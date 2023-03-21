from PIL import Image, ImageOps
import os
import numpy as np

VALID_FORMATS = (".jpeg", ".jpg", ".png")
SAMPLE_SIZE = 32
MAX_RESOLUTION = (1080, 1080)
REDUCTION_FACTOR = 4


def resize_image(img):
    width, height = img.size
    max_width, max_height = MAX_RESOLUTION
    if (width > max_width) or (height > max_height):
        img = img.resize((width // REDUCTION_FACTOR, height // REDUCTION_FACTOR))
    return img


def determine_max_valid_index(sample_size, offset, dimension_len):
    indices = [i for i in range(0, dimension_len, offset)]
    i = -1
    while (indices[i] + sample_size) > dimension_len:
        i -= 1
    return indices[i]


def save_letter(image_matrix, name):
    image = Image.fromarray(image_matrix)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    file_path = f"{name}.jpg"
    image.save(file_path)


def sampler(image):
    image_sample = []
    temp_array = []
    for column in range(0, len(image[0]), 8):
        for row in range(0, len(image[0]), 8):
            image_sample.append(image[column:column + 8, row:row + 8])
            temp_array.append(np.array([num for sublist in image_sample[-1] for num in sublist]))
    return temp_array


def rebuild_32x32(letter_data, sample_index):
    letter_data = np.reshape(letter_data, (16, 64))

    temp_array = []

    for i in range(len(letter_data)):
        temp_array.append(np.reshape(letter_data[i], (8, 8)))

    temp_image = np.block([[temp_array[0], temp_array[1], temp_array[2], temp_array[3]],
                              [temp_array[4], temp_array[5], temp_array[6], temp_array[7]],
                              [temp_array[8], temp_array[9], temp_array[10], temp_array[11]],
                              [temp_array[12], temp_array[13], temp_array[14], temp_array[15]]])

    #temp_image = numpy.reshape(letter_data[0], (8, 8))
    print(temp_image)
    save_letter(temp_image, "ImageTestingOuput/image" + str(sample_index))


def multi_image_processor(offset=4, save_image=False):
    # image paths list
    image_paths = []
    file_names = []

    # walk 'images_to_process' folder to find images to work on
    for (root, dirs, files) in os.walk('images_to_process'):
        for file in files:
            if file.endswith(VALID_FORMATS):
                image_paths.append(os.path.join(root, file))
                file_names.append(file.split(".")[0])

    # loop through found images/paths
    for image_count, image_path in enumerate(image_paths):
        image_data_8x8 = []
        with Image.open(image_path) as img:
            # get image dimensions, grayscale, load pixel data
            img = resize_image(img)
            width, height = img.size
            img = ImageOps.grayscale(img)
            img_pixel_data = img.load()
            # boost contrast
            for i_index in range(height):
                for j_index in range(width):
                    if img_pixel_data[j_index, i_index] > 110:
                        img_pixel_data[j_index, i_index] = 255
                    else:
                        img_pixel_data[j_index, i_index] = 0

        if save_image:
            # get file name
            file_name = image_path.split(".")[0]
            file_name = file_name.split("\\")[1]
            img.save(f"{file_name}.jpg")

        # determine max index to take sample within image dimensions
        max_width_index = determine_max_valid_index(SAMPLE_SIZE, offset, width)
        max_height_index = determine_max_valid_index(SAMPLE_SIZE, offset, height)

        # take 32x32 sample in 8x8 pieces for each offset step
        for i in range(0, max_height_index + 1, offset):
            print(f"{file_names[image_count]} : row {i} of {max_height_index}\r", end="")
            for j in range(0, max_width_index + 1, offset):
                sample_32x32 = []
                # build 32x32 sample
                for y_index in range(i, i + SAMPLE_SIZE):
                    temp_array = []
                    for x_index in range(j, j + SAMPLE_SIZE):
                        temp_array.append(img_pixel_data[x_index, y_index])
                    sample_32x32.append(temp_array)
                # cut sample into 16 8x8 pieces and add to 8x8 image data
                pieces_8x8 = sampler(np.array(sample_32x32))
                for piece in pieces_8x8:
                    image_data_8x8.append(piece)

        # convert to numpy array and save
        image_data_8x8 = np.array(image_data_8x8)
        print(f"\n{file_names[image_count]} : {image_data_8x8.shape}")
        np.save(f"{file_names[image_count]}_8x8_data", image_data_8x8)


def single_image_processor(offset=4, image_path=None, save_file=False):
    # validation
    if not image_path:
        print("Image not specified")
        exit()

    if not image_path.endswith(VALID_FORMATS):
        print("Image not valid format")
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
        # boost contrast
        for i_index in range(height):
            for j_index in range(width):
                if img_pixel_data[j_index, i_index] > 110:
                    img_pixel_data[j_index, i_index] = 255
                else:
                    img_pixel_data[j_index, i_index] = 0

    # determine max index to take sample within image dimensions
    max_width_index = determine_max_valid_index(SAMPLE_SIZE, offset, width)
    max_height_index = determine_max_valid_index(SAMPLE_SIZE, offset, height)

    # take 32x32 sample in 8x8 pieces for each offset step
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
            # cut sample into 16 8x8 pieces and add to 8x8 image data
            pieces_8x8 = sampler(np.array(sample_32x32))
            for piece in pieces_8x8:
                image_data_8x8.append(piece)

    # convert to numpy array
    image_data_8x8 = np.array(image_data_8x8)
    print(f"\n{file_name} : {image_data_8x8.shape}")

    if save_file:
        np.save(f"{file_name}_8x8_data_single", image_data_8x8)

    print("image data 8x8 shape: ", np.shape(image_data_8x8))
    return offset, image_data_8x8, width, height, img_pixel_data


if __name__ == "__main__":
    offset, image_data = multi_image_processor(save_image=True)
    # print(offset)