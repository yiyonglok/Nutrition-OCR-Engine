import numpy
import ImageProcessor


# read in letter and non-letter images, and then put together and save
# to unshuffled unlabelled dataset
# and unshuffled labelled dataset
if __name__ == "__main__":
    letters = ImageProcessor.ReadLetterImages("Char74k_32x32")
    nonletters = ImageProcessor.ReadNonLetterImages("non_letters")

    letters_length = len(letters)
    nonletters_length = len(nonletters)

    unshuffled_data = numpy.concatenate((letters, nonletters), axis=0)

    numpy.save("unshuffled_unlabeled_data", unshuffled_data)

    class_one = numpy.full((letters_length, 1), 1)
    class_two = numpy.full((nonletters_length, 1), 0)
    classes = numpy.concatenate((class_one, class_two), axis=0)

    unshuffled_data = numpy.concatenate((unshuffled_data, classes), axis=1)

    numpy.save("unshuffled_labeled_data", unshuffled_data)