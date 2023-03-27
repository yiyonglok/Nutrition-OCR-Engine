import numpy
import skimage.io
from PIL import Image, ImageOps
import os
import platform
import matplotlib.pyplot as plt

MAX_RESOLUTION = (24, 24)
REDUCTION_FACTOR = 8


def get_image_paths(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file or '.jpg' in file:
                files.append(os.path.join(r, file))
    return files

def image_name_parser(filename):
    if "mac" in platform.system().lower() or "linux" in platform.system().lower():
        print("mac")
        file_path = filename.split("/")
    else:
        print("windows")
        file_path = filename.split("\\")
    return int(file_path[2][3:6])


def read_letter_images(filepath):
    load_letters = get_image_paths(filepath)

    data_array = []
    progress = 0
    for file in load_letters:
        print(f"Progress: {progress*100/len(load_letters)}%")
        if filepath != "non_letters":
            classifier = image_name_parser(file)
        else:
            classifier = 0

        letter = skimage.util.img_as_ubyte(skimage.io.imread(file, as_gray=True))

        if len(data_array) == 0:
            data_array = numpy.array(sampler(letter, classifier))
        else:
            data_array = numpy.concatenate((data_array, sampler(letter, classifier)), axis=0)
        progress += 1

    data_array = numpy.array(data_array)

    return data_array

def image_resize(image_path):
    image = Image.open(image_path)
    new_image = image.resize((32,32))
    new_image.save(image_path)


def image_whitener(file):
    nonletter = skimage.io.imread(file, as_gray=True)
    print("Image pixel value mean:", numpy.mean(nonletter))
    nonletter = nonletter + numpy.mean(nonletter)
    image = Image.fromarray(nonletter)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    new_file_path = file.split("\\")
    new_file = new_file_path[-1].split(".")
    s = "\\"
    new_file_path = s.join(new_file_path[:-1])
    new_file_path = new_file_path + "\\" + str(new_file[0]) + "_whitened.jpg"
    print(new_file_path)
    image.save(new_file_path)


def sampler(image, classifier):
    image_sample = []
    temp_array = []
    for column in range(0, len(image[0]), 8):
        for row in range(0, len(image[0]), 8):
            image_sample.append(image[column:column + 8, row:row + 8])
            temp_array.append(numpy.concatenate(
                (numpy.array([num for sublist in image_sample[-1] for num in sublist]), [classifier])))
    return temp_array


def delete_bad_image(file):
    print(file)
    if "whitened.jpg" in file:
        os.remove(file)


def read_nonletter_images(filepath):
    load_nonletters = get_image_paths(filepath)

    data_array = []
    count = 0
    for file in load_nonletters:
        print("Progress:", str(count * 100 / len(load_nonletters)), "%")

        classifier = 0

        nonletter = skimage.util.img_as_ubyte(skimage.io.imread(file, as_gray=True))

        if len(data_array) == 0:
            data_array = numpy.array(sampler(nonletter, classifier))
        else:
            data_array = numpy.concatenate((data_array, sampler(nonletter, classifier)), axis=0)
        count += 1

    return data_array


def add_border(image):
    size = len(image)
    filler = numpy.full((size, int((256 - size) / 2)), 255)
    image = numpy.concatenate((filler, image), axis=1)
    image = numpy.concatenate((image, filler), axis=1)

    filler = numpy.full((int((256 - size) / 2), 256), 255)
    image = numpy.concatenate((filler, image), axis=0)
    image = numpy.concatenate((image, filler), axis=0)
    return image


def average_pooling(image):
    print(len(image))
    # use max pooling to shrink image
    new_image = []
    for i in range(0, len(image), 2):
        row = []
        for j in range(0, len(image[i]), 2):
            calculate_avg = image[i][j] + image[i + 1][j] + image[i][j + 1] + image[i + 1][j + 1]
            calculate_avg = calculate_avg / 4
            row.append(calculate_avg)
        new_image.append(row)
    print(len(image))
    return numpy.array(new_image)


def shrink_lowercase_samples(file_path):
    files = get_image_paths(file_path)

    for file in files:
        with Image.open(file) as img:
            print(img)
            # get image dimensions, grayscale, load pixel data
            img = resize_image(img)
            # letter = numpy.array(img)
            img.save(f"{file}")


def resize_image(img):
    width, height = 32, 32
    max_width, max_height = MAX_RESOLUTION
    if (width > max_width) or (height > max_height):
        img = img.resize((width - REDUCTION_FACTOR, height - REDUCTION_FACTOR))
    return img


def add_lowercase_border(file_path):
    files = get_image_paths(file_path)

    for file in files:
        small_image = skimage.io.imread(file, as_gray=True)
        small_image = numpy.array(small_image)
        border_size = int(REDUCTION_FACTOR / 2)
        filler = numpy.full((int(border_size), 32 - REDUCTION_FACTOR), 255)
        small_image = numpy.concatenate((filler, small_image), axis=0)
        small_image = numpy.concatenate((small_image, filler), axis=0)
        filler = numpy.full((32, (int(border_size))), 255)
        small_image = numpy.concatenate((filler, small_image), axis=1)
        small_image = numpy.concatenate((small_image, filler), axis=1)
        save_letter(small_image, file)


def save_letter(image_matrix, name):
    image_matrix = numpy.array(image_matrix)
    image = Image.fromarray(image_matrix)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    file_path = f"{name}"
    image.save(file_path)


if __name__ == "__main__":
    None
    #shrink_lowercase_samples("Char74k_32x32_cleaned_deanna")
    #add_lowercase_border("Char74k_32x32_cleaned_deanna")