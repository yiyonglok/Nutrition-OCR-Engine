import numpy

labels_as_letters = {1:"0", 2:"1", 3:"2", 4:"3", 5:"4", 6:"5", 7:"6", 8:"7", 9:"8", 10:"9",
                11:"A", 12:"B", 13:"C", 14:"D", 15:"E", 16:"F", 17:"G", 18:"H", 19:"I", 20:"J",
                21:"K", 22:"L", 23:"M", 24:"N", 25:"O", 26:"P", 27:"Q", 28:"R", 29:"S", 30:"T",
                31:"U", 32:"V", 33:"W", 34:"X", 35:"Y", 36:"Z", 37:"a", 38:"b", 39:"c", 40:"d",
                41:"e", 42:"f", 43:"g", 44:"h", 45:"i", 46:"j", 47:"k", 48:"l", 49:"m", 50:"n",
                51:"o", 52:"p", 53:"q", 54:"r", 55:"s", 56:"t", 57:"u", 58:"v", 59:"w", 60:"x",
                61:"y", 62:"z"}


def translate_letters_to_words(letters):
    if isinstance(letters, str):
        # with open(letters, 'rb') as opened_file:
        #    letter_data = numpy.load(letters)
        letter_data = open(letters, "r")
        letter_data = letter_data.read()
        letter_data = parse_text_to_array(letter_data)
    else:
        letter_data = letters

    word = ""
    letter_counter = 0
    for i in range(len(letter_data)):
        if i == 0:
            letter_counter += 1
        else:
            if letter_data[i] == letter_data[i - 1]:
                letter_counter += 1
            else:
                letter_counter = 1
        if letter_counter > 0 and letter_counter % 2 == 0:
            word += letter_data[i]

    return word


def parse_text_to_array(letters):
    letters = letters.replace('[', '')
    letters = letters.replace(']', '')
    letters = letters.replace(' ', '')
    letters = letters.replace("''", "'")

    letter_data = letters.split("'")
    letter_data = letter_data[1:-1]

    return letter_data


if __name__ == "__main__":
    translate_letters_to_words("LetterRecognitionsLetterFiltered.txt")