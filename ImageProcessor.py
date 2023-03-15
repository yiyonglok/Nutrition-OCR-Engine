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
    return files

def image_name_parser(filename):
    file_path = filename.split("\\")
    return int(file_path[2][3:6])


def read_letter_images(filepath):
    load_letters = get_image_paths(filepath)

    #load_letters = load_letters[:10]
    print(len(load_letters))

    data_array = []
    progress = 0
    for file in load_letters:
        print(f"Progress: {progress*100/len(load_letters)}%")
        if filepath != "non_letters":
            classifier = image_name_parser(file)
        else:
            classifier = 0

        #print(classifier)
        letter = skimage.io.imread(file, as_gray=True)
        if len(data_array) == 0:
            data_array = numpy.array(sampler(letter, classifier))
        else:
            data_array = numpy.concatenate((data_array, sampler(letter, classifier)), axis=0)
        progress += 1

    data_array = numpy.array(data_array)
    # print(data_array[-1])

    # numpy.save("Unshuffled_datafile", data_array)
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
    new_file_path = new_file_path + "\\" +str(new_file[0]) + "_whitened.jpg"
    print(new_file_path)
    image.save(new_file_path)


def sampler(image, classifier):
    image_sample = []
    temp_array = []
    for column in range(0, len(image[0]), 8):
        for row in range(0, len(image[0]), 8):
            image_sample.append(image[column:column + 8, row:row + 8])
            temp_array.append(numpy.concatenate((numpy.array([num for sublist in image_sample[-1] for num in sublist]), [classifier])))
    return temp_array


def delete_bad_image(file):
    print(file)
    if "whitened.jpg" in file:
        print("yes")
        os.remove(file)


def read_nonletter_images(filepath):
    load_nonletters = get_image_paths(filepath)
    print(len(load_nonletters))
    #load_nonletters = load_nonletters[:2]

    data_array = []
    count = 0
    for file in load_nonletters:
        print("Progress:", str(count*100/len(load_nonletters)), "%")

        classifier = 0

        #print(file)
        nonletter = skimage.io.imread(file, as_gray=True)

        if len(data_array) == 0:
            data_array = numpy.array(sampler(nonletter, classifier))
        else:
            data_array = numpy.concatenate((data_array, sampler(nonletter, classifier)), axis=0)
        count += 1

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


if __name__ == "__main__":
    #letters = read_letter_images()
    nonletters = read_nonletter_images()

    #letters_length = len(letters)
    nonletters_length = len(nonletters)

    #print(len(letters))
    print(len(nonletters))

    #unshuffled_data = numpy.concatenate((letters, nonletters), axis=0)
    numpy.save("nonletter_data_array.npy", nonletters)

    #class_one = numpy.full((letters_length, 1), 1)
    #class_two = numpy.full((nonletters_length, 1), 0)
    #classes = numpy.concatenate((class_one, class_two), axis=0)

    #unshuffled_data = numpy.concatenate((unshuffled_data, classes), axis=1)
    #numpy.save("unshuffled_labeled_data", unshuffled_data)