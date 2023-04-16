import numpy
import ImageProcessor

def ProcessTrainingImages(file_name, saved_file_name):
    unshuffled_letter_data = ImageProcessor.read_letter_images(file_name)
    numpy.save(saved_file_name, unshuffled_letter_data)



# if __name__ == "__main__":
#     unshuffled_letter_data = ImageProcessor.read_letter_images("Char74k_32x32_cleaned_faizan")
#     numpy.save("unshuffled_letter_data_cleaned_faizan", unshuffled_letter_data)
#
#     nonletter_data_array = ImageProcessor.read_nonletter_images("non_letters")
#     numpy.save("nonletter_data_array.py", nonletter_data_array)