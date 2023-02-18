

from PIL import Image
import math
import numpy


class HeatMap:
    def __init__(self, image_width, image_height, samples_count, offset):
        self.offset = offset
        self.samples_count = samples_count
        self.image_width = image_width
        self.image_height = image_height
        self.image = numpy.full((self.image_height, self.image_width), 0)


    def update_heat_map(self, sample_index):
        sample_width = ( self.image_width - 32 )/ self.offset
        sample_height = self.samples_count / sample_width

        row = int( math.trunc(sample_index / sample_width ) * self.offset)
        column = int( (sample_index - (row * sample_width / self.offset )) * self.offset  )

        #print("sample_width:", sample_width)
        #print("sample_height", sample_height)
        #print("row", row)
        #print("column", column)
        #print(self.image[row:row+32, column:column+32])
        self.image[row:row+32, column:column+32] = self.image[row:row+32, column:column+32] + 5


    def print_heat_map(self):
        image = Image.fromarray(self.image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save("heatmap.jpg")


    def print_dimensions(self):
        print(str(len(self.image[0])) + "x" + str(len(self.image)))