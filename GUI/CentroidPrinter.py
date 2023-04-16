import numpy
from PIL import Image
from skimage.io import imread_collection


# pass file containing centroid data
def SaveCentroidImages(file):
    centroid_data = numpy.load(file)

    for i in range(len(centroid_data)):
        centroid_reshaped = numpy.reshape(centroid_data[i], (8, 8))

        image = Image.fromarray(centroid_reshaped)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save("centroid_images/centroid_" + str(i) + ".jpg")


def SaveCentroidCollage(file, number_of_centroids):

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

    centroid_data = numpy.load(file)

    centroid_collage = []
    centroid_horizontal = []
    for i in range(len(centroid_data)):
        if i % length == 0:
            centroid_horizontal = numpy.reshape(centroid_data[i], (8, 8))
        else:
            centroid_reshaped = numpy.reshape(centroid_data[i], (8, 8))
            centroid_horizontal = numpy.hstack((centroid_horizontal, centroid_reshaped))

        if i == length - 1:
            centroid_collage = centroid_horizontal
        elif i != 0 and i % length == length - 1:
            centroid_collage = numpy.vstack((centroid_collage, centroid_horizontal))
            centroid_horizontal = []

    image = Image.fromarray(centroid_collage)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save("collage_" + str(number_of_centroids) + ".jpg")


SaveCentroidCollage('centroid_data.npy', 64)