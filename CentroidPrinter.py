import numpy
from PIL import Image


# pass file containing centroid data
def SaveCentroidImages(file):
    centroid_data = numpy.load(file)

    for i in range(len(centroid_data)):
        centroid_reshaped = numpy.reshape(centroid_data[i], (8, 8))

        image = Image.fromarray(centroid_reshaped)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save("centroid_" + str(i) + ".jpg")
