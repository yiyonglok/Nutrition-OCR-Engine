import numpy
import skimage.io
import os

def get_image_paths(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))
    #for f in files:
    #    print(f)
    return files

def image_name_parser(filename):
    file_path = filename.split("\\")
    #print(file_path[2][3:6])
    return int(file_path[2][3:6])

if __name__ == "__main__":
    load_letters = get_image_paths("Char74k_32x32")

    #load_letters = load_letters[:1100]
    #print(load_letters)

    data_array = []

    for file in load_letters:
        print(file)
        letter = skimage.io.imread(file, as_gray=True)

        letter_sample = []
        for column in range(0,len(letter[0]),8):
            for row in range(0, len(letter[0]), 8):
                letter_sample.append(letter[column:column+8,row:row+8])
                temp_array = numpy.array([num for sublist in letter_sample[-1] for num in sublist])
                temp_array = numpy.append(temp_array, image_name_parser(file))
                data_array.append(temp_array)


    data_array = numpy.array(data_array)

    print(data_array[-1])

    numpy.save("Unshuffled_datafile", data_array)