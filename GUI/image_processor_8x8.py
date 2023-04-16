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


def resize_image_by_height(img):

    letter_height_endpoints = []
    img_pixel_data = img.load()
    width, height = img.size

    # boost contrast
    #for i_index in range(height):
    #    for j_index in range(width):
    #        if img_pixel_data[j_index, i_index] > 110:
    #            img_pixel_data[j_index, i_index] = 255
    #        else:
    #            img_pixel_data[j_index, i_index] = 0

    sample_32x32 = []
    # build 32x32 sample
    for y_index in range(0, height):
        temp_array = []
        for x_index in range(0, width):
            temp_array.append(img_pixel_data[x_index, y_index])
        sample_32x32.append(temp_array)

    img_pixel_data = np.array(sample_32x32)

    # find top and bottom of letter
    is_space = False
    for i in range(int(height / 2), -1, -1):
        if not is_space and np.count_nonzero(img_pixel_data[i, :] == 0) < 10:
            is_space = True
            letter_height_endpoints.append(i)
        if is_space:
            break
    if not is_space:
        letter_height_endpoints.append(0)

    is_space = False
    for i in range(int(height / 2), height):
        if not is_space and np.count_nonzero(img_pixel_data[i, :] == 0) < 10:
            is_space = True
            letter_height_endpoints.append(i)
        if is_space:
            break
    if not is_space:
        letter_height_endpoints.append(height)

    letter_size = letter_height_endpoints[1] - letter_height_endpoints[0]
    print("size:", letter_size)

    if letter_size > 25:
        aspect_ratio = width / height
        height_reduction = int(height - (letter_size - 25))
        if height_reduction < 33:
            height_reduction = 33
        width_reduction = int(aspect_ratio * height_reduction)
        img = img.resize((width_reduction, height_reduction))
        print(height_reduction)

    return img


def determine_max_valid_index(sample_size, offset, dimension_len):
    indices = [i for i in range(0, dimension_len, offset)]
    i = -1
    while (indices[i] + sample_size) > dimension_len - 1:
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


def is_quality_sample(image):
    if np.min(image) > 200:
        return False
    min_first_col = np.min(image[:, 0])
    min_last_col = np.min(image[:, -1])
    min_first_row = np.min(image[0, :])
    min_last_row = np.min(image[-1, :])
    if (min_first_col < 5) != (min_last_col < 5): #and (np.count_nonzero(image[:, 0] == min_first_col) > 5 or np.count_nonzero(image[:, 0] == min_last_col) > 5):
        return False
    if (min_first_row < 5) != (min_last_row < 5): #and (np.count_nonzero(image[:, 0] == min_first_row) > 5 or np.count_nonzero(image[:, 0] == min_last_row) > 5):
        return False
    return True


def dynamically_crop_image(image):
    image = np.array(image)
    width = len(image)
    b = np.min(image, axis=0)

    letter_width_endpoints = []
    letter_height_endpoints = []

    # crops sides
    is_space = False
    for i in range(int(width/ 2), -1, -1):
        if not is_space and b[i] > 5:
            is_space = True
            image[:, :i] = 255
            letter_width_endpoints.append(i)
        if is_space:
            break
    #if not is_space:
    #    letter_width_endpoints.append(0)

    is_space = False
    for i in range(int(width / 2), width):
        if not is_space and b[i] > 5:
            is_space = True
            image[:, i:] = 255
            letter_width_endpoints.append(i)
        if is_space:
            break
    #if not is_space:
    #    letter_width_endpoints.append(32)

    d = np.min(image, axis=1)

    # crop top and bottom
    is_space = False
    for i in range(int(width / 2), -1, -1):
        if not is_space and d[i] > 5:
            is_space = True
            image[:i, :] = 255
            letter_height_endpoints.append(i)
        if is_space:
            break
    #if not is_space:
    #    letter_height_endpoints.append(0)

    is_space = False
    for i in range(int(width / 2), width):
        if not is_space and d[i] > 5:
            is_space = True
            image[i:, :] = 255
            letter_height_endpoints.append(i)
        if is_space:
            break
    #if not is_space:
    #    letter_height_endpoints.append(32)

    if len(letter_height_endpoints) == 2:
        shift = letter_height_endpoints[0] - (width - letter_height_endpoints[1])
        #if abs(shift) > 15:
        #    None
        if shift < -1:
            shift = abs(shift)
            new_column = np.zeros((int(shift/2), width)) + 255
            image = image[:-int(shift/2), :]
            image = np.concatenate((new_column, image), axis=0)

        elif shift > 1:
            new_column = np.zeros((int(shift/2), width)) + 255
            image = image[int(shift/2):, :]
            image = np.concatenate((image, new_column), axis=0)

    if len(letter_width_endpoints) == 2:
        shift = letter_width_endpoints[0] - (width - letter_width_endpoints[1])
        #if abs(shift) > 15:
        #    None
        if shift < -1:
            shift = abs(shift)
            new_column = np.zeros((width, int(shift / 2))) + 255
            image = image[:, :-int(shift / 2)]
            image = np.concatenate((new_column, image), axis=1)

        elif shift > 1:
            new_column = np.zeros((width, int(shift / 2))) + 255
            image = image[:, int(shift / 2):]
            image = np.concatenate((image, new_column), axis=1)

    #print(letter_width_endpoints)
    #print("width of letter?", letter_width_endpoints[1] - letter_width_endpoints[0])
    #print("height of letter?", letter_height_endpoints[1] - letter_height_endpoints[0])
    #print("image shape now: ", np.shape(image))

    return image


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
            width, height = img.size
            img = ImageOps.grayscale(img)
            img = resize_image(img)
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


def single_image_processor(offset=4, image_path=None, save_file=False, crop_image=False, remove_bad_samples=False, resize_by_height=False):
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
        if resize_by_height:
            img = resize_image_by_height(img)
            width, height = img.size
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
            if crop_image:
                sample_32x32 = dynamically_crop_image(sample_32x32)
                #sample_32x32 = dynamically_crop_image(sample_32x32)
            else:
                sample_32x32 = np.array(sample_32x32)
            if remove_bad_samples:
                if not is_quality_sample(sample_32x32):
                    sample_32x32[:, :] = 255
            # save_letter(sample_32x32, f"reshaped_images/{count}")
            count += 1
            # cut sample into 16 8x8 pieces and add to 8x8 image data
            pieces_8x8 = sampler(np.array(sample_32x32))
            for piece in pieces_8x8:
                image_data_8x8.append(piece)

    # convert to numpy array
    image_data_8x8 = np.array(image_data_8x8)
    print(f"\n{file_name} : {image_data_8x8.shape}")

    if save_file:
        np.save(f"myfile", image_data_8x8)

    print("image data 8x8 shape: ", np.shape(image_data_8x8))
    return offset, image_data_8x8, width, height, img_pixel_data, max_height_index, max_width_index

#
# if __name__ == "__main__":
#     offset, image_data = multi_image_processor(save_image=True)
#     # print(offset)