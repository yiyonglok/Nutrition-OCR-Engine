import numpy
import skimage.io
import os


# get images from within the project folder
def GetImagesByFileName(file_name):
    files = []
    for r, d, f in os.walk(file_name):
        for file in f:
            if '.png' in file or '.jpg' in file:
                files.append(os.path.join(r, file))
    return files


# read images that are letters, do not need re-sizing
def ReadLetterImages(file_name):
    load_letters = GetImagesByFileName(file_name)

    data_array = []
    for file in load_letters:
        letter = skimage.io.imread(file, as_gray=True)
        if len(data_array) == 0:
            data_array = numpy.array(Sample(letter))
        else:
            data_array = numpy.concatenate((data_array, Sample(letter)), axis=0)

    data_array = numpy.array(data_array)
    return data_array


# read images that are not letters, need resizing
def ReadNonLetterImages(file_name):
    load_nonletters = GetImagesByFileName(file_name)

    data_array = []
    for file in load_nonletters:
        nonletter = skimage.io.imread(file, as_gray=True)
        nonletter = AddBorder(nonletter)
        while len(nonletter) > 32:
            nonletter = AveragePool(nonletter)
        if len(data_array) == 0:
            data_array = numpy.array(Sample(nonletter))
        else:
            data_array = numpy.concatenate((data_array, Sample(nonletter)), axis=0)
    return data_array


# specific to file names with format _______
def ParseImageName(file_name):
    file_path = file_name.split("\\")
    return int(file_path[2][3:6])


# helper method to divide images into 8x8 samples
def Sample(image):
    image_sample = []
    temp_array = []
    for column in range(0, len(image[0]), 8):
        for row in range(0, len(image[0]), 8):
            image_sample.append(image[column:column + 8, row:row + 8])
            temp_array.append(numpy.array([num for sublist in image_sample[-1] for num in sublist]))
    return temp_array


# resize images to 32x32 format by adding pixels to the border
def AddBorder(image):
    size = len(image)
    zeros = numpy.zeros((size, int((256 - size) / 2)))
    image = numpy.concatenate((zeros, image), axis=1)
    image = numpy.concatenate((image, zeros), axis=1)

    zeros = numpy.zeros((int((256 - size) / 2), 256))
    image = numpy.concatenate((zeros, image), axis=0)
    image = numpy.concatenate((image, zeros), axis=0)
    return image


# shrink images by taking average value of 4 pixels and combining into 1 pixel
def AveragePool(image):
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