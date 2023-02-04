import numpy
import skimage.io
from PIL import Image
import os

def get_image_paths(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file or '.jpg' in file:
                files.append(os.path.join(r, file))
    #files = files[0:10]
    #for f in files:
    #    print(f)
    return files

def image_name_parser(filename):
    file_path = filename.split("\\")
    #print(file_path[2][3:6])
    return int(file_path[2][3:6])


def read_letter_images():
    load_letters = get_image_paths("Char74k_32x32")

    #load_letters = load_letters[:10]
    print(len(load_letters))

    data_array = []

    for file in load_letters:
        #print(file)
        letter = skimage.io.imread(file, as_gray=True)
        if len(data_array) == 0:
            data_array = numpy.array(sampler(letter))
        else:
            data_array = numpy.concatenate((data_array, sampler(letter)), axis=0)

    data_array = numpy.array(data_array)
    # print(data_array[-1])

    # numpy.save("Unshuffled_datafile", data_array)

    return data_array


def sampler(image):
    image_sample = []
    temp_array = []
    for column in range(0, len(image[0]), 8):
        for row in range(0, len(image[0]), 8):
            image_sample.append(image[column:column + 8, row:row + 8])
            temp_array.append(numpy.array([num for sublist in image_sample[-1] for num in sublist]))
    return temp_array


def read_nonletter_images():
    load_nonletters = get_image_paths("non_letters")
    print(len(load_nonletters))
    #load_nonletters = load_nonletters[:5000]

    data_array = []
    for file in load_nonletters:
        #print(file)
        nonletter = skimage.io.imread(file, as_gray=True)
        nonletter = add_border(nonletter)
        while len(nonletter) > 32:
            nonletter = average_pooling(nonletter)
        if len(data_array) == 0:
            data_array = numpy.array(sampler(nonletter))
        else:
            data_array = numpy.concatenate((data_array, sampler(nonletter)), axis=0)
    return data_array

    #print(len(data_array))
    #print(len(data_array[0]))
    #print(len(data_array))

    # iimage = Image.fromarray(data_array[0])
    # if iimage.mode != 'RGB':
    #     iimage = iimage.convert('RGB')
    # iimage.save("test-pic-2.jpg")


def add_border(image):
    size = len(image)
    zeros = numpy.zeros((size, int((256 - size) / 2)))
    image = numpy.concatenate((zeros, image), axis=1)
    image = numpy.concatenate((image, zeros), axis=1)

    zeros = numpy.zeros((int((256 - size) / 2), 256))
    image = numpy.concatenate((zeros, image), axis=0)
    image = numpy.concatenate((image, zeros), axis=0)
    return image


def average_pooling(image):

    # use max pooling to shrink image
    new_image = []
    for i in range(0, len(image), 2):
        row = []
        for j in range(0, len(image[i]), 2):
            calculate_avg = image[i][j] + image[i + 1][j] + image[i][j + 1] + image[i + 1][j + 1]
            calculate_avg = calculate_avg / 4
            row.append(calculate_avg)
        new_image.append(row)

    return numpy.array(new_image)


if __name__ == "__main__":
    letters = read_letter_images()
    nonletters = read_nonletter_images()

    letters_length = len(letters)
    nonletters_length = len(nonletters)

    print(len(letters))
    print(len(nonletters))

    unshuffled_data = numpy.concatenate((letters, nonletters), axis=0)

    numpy.save("unshuffled_unlabeled_data", unshuffled_data)

    class_one = numpy.full((letters_length, 1), 1)
    class_two = numpy.full((nonletters_length, 1), 0)
    classes = numpy.concatenate((class_one, class_two), axis=0)

    unshuffled_data = numpy.concatenate((unshuffled_data, classes), axis=1)

    numpy.save("unshuffled_labeled_data", unshuffled_data)