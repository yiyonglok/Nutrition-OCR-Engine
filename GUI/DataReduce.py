import numpy
import skimage.io
from PIL import Image
import os


def CollectData(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file or '.jpg' in file:
                files.append(os.path.join(r, file))
    return files

def CollectRefNumber(load_letters):
    data_array = []

    for file in load_letters:
        data_array += [(image_name_parser(file))]

    return data_array

def image_name_parser(filename):
    file_path = filename.split('/')
    return file_path[2][7:12]

def RemoveFiles(filenames,refvalues):
    print(refvalues[0])
    print(filenames[0][41:46])

    for name in filenames:
        for val in range(0,len(refvalues)):
            if name[41:46] == refvalues[val]:
                break
            elif val == len(refvalues)-1:
                if os.path.exists(name):
                    os.remove(name)
                else:
                    print("The file does not exist")

