import numpy
from PIL import Image


def save_centroid_image(centroid, label):
    image = Image.fromarray(centroid)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save("centroid_" + label + ".jpg")


centroid_data = numpy.load("centroid_data.npy")
#print(centroid_data)
#print(len(centroid_data))

for i in range(len(centroid_data)):
    centroid_reshaped = numpy.reshape(centroid_data[i], (8, 8))
    #print(centroid_data[i])
    #print(centroid_reshaped)
    #print("\n")

    save_centroid_image(centroid_reshaped, str(i))