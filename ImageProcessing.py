import numpy
import ImageProcessor


# read in letter and non-letter images, and then put together and save
# to unshuffled unlabelled dataset
# and unshuffled labelled dataset
if __name__ == "__main__":
    unshuffled_letter_data = ImageProcessor.read_letter_images("Char74k_32x32_cleaned_faizan")
    numpy.save("unshuffled_letter_data_cleaned_faizan", unshuffled_letter_data)

    nonletter_data_array = ImageProcessor.read_nonletter_images("non_letters")
    numpy.save("nonletter_data_array.py", nonletter_data_array)