from PIL import Image
import os
import random
import numpy as np


def printData(filepath,destination):

    listofimages = []
    for x in range(100):
        sampleimg = random.choice(os.listdir(filepath))
        if sampleimg == ".DS_Store":
            sampleimg = "Sample001"

        listofimages.append(filepath + "/" + sampleimg + "/" + random.choice(os.listdir(filepath + "/" + sampleimg)))

    collage_maker(listofimages,destination)

def nonLetterCollage(filepath, destination):

    new = Image.new("RGBA", (320,320))

    for x in range(100):
        choice = random.choice(os.listdir(filepath))
        if choice == ".DS_Store":
            choice = random.choice(os.listdir(filepath))
        img = Image.open(filepath + "/" + choice)
        row = (x // 10) * 32
        column = (x % 10) * 32
        new.paste(img, (row, column))
        print("we are at row " + str(row) + " and column " + str(column))

    new.save(destination)

def centroidCollage(file, number_of_centroids, destination):

    # find divisors to determine length and width of collage
    divisors = []
    for i in range(1, number_of_centroids + 1):
        if number_of_centroids % i == 0:
            divisors.append(i)
            print(i)
    print(divisors)

    length_divisors = len(divisors)
    if length_divisors % 2 == 1:
        length_divisors = length_divisors - 1
    index = int(length_divisors / 2)

    length = divisors[index]

    centroid_data = np.load(file)

    centroid_collage = []
    centroid_horizontal = []
    for i in range(len(centroid_data)):
        if i % length == 0:
            centroid_horizontal = np.reshape(centroid_data[i], (8, 8))
        else:
            centroid_reshaped = np.reshape(centroid_data[i], (8, 8))
            centroid_horizontal = np.hstack((centroid_horizontal, centroid_reshaped))

        if i == length - 1:
            centroid_collage = centroid_horizontal
        elif i != 0 and i % length == length - 1:
            centroid_collage = np.vstack((centroid_collage, centroid_horizontal))
            centroid_horizontal = []

    image = Image.fromarray(centroid_collage)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(destination)



def collage_maker(listofimages, destination):

    new = Image.new("RGBA", (320,320))

    for x in range(100):
        img = Image.open(listofimages[x])
        row = (x // 10) * 32
        column = (x % 10) * 32
        new.paste(img, (row,column))
        print("we are at row " + str(row) + " and column " + str(column))

    new.save(destination)

def SaveCentroidImages(file):
    centroid_data = np.load(file)

    for i in range(len(centroid_data)):
        centroid_reshaped = np.reshape(centroid_data[i], (8, 8))

        image = Image.fromarray(centroid_reshaped)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save("SavedCentroidImages/centroid_" + str(i) + ".jpg")

def image_resize(image_path):
    image = Image.open(image_path)
    new_image = image.resize((320,320))
    new_image.save(image_path)








