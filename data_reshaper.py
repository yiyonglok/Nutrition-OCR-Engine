import numpy as np
from PIL import Image

def save_letter(letter, count):
    image = Image.fromarray(letter)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    file_path = f"reshaped_images/image_{count}.jpg"
    image.save(file_path)


def reshape_letter_data(letter_data, save_images=False, save_file=False):
    reshaped_letter_data = []

    for count, image in enumerate(letter_data):
        #init blank 32x32 array
        reshaped_image = [[0] * 32 for i in range(32)]

        #determine coords of where to place each 8x8 piece in 32x32 array
        coords = []
        for y in range(0, 32, 8):
            for x in range(0, 32, 8):
                coords.append((x, y))

        #get each 8x8 piece
        pieces = []
        for i in range(0, 1024, 64):
            pieces.append(image[i:i+64])

        #place each piece according to coords in 32x32 array
        for piece, coord in enumerate(coords):
            x, y = coord
            letter_data_index = -1
            for y_index in range(8):
                for x_index in range(8):
                    letter_data_index += 1
                    reshaped_image[y+y_index][x+x_index] = pieces[piece][letter_data_index]
        
        #flatten image
        flattened_image = []
        for y in range(len(reshaped_image)):
            for x in range(len(reshaped_image[0])):
                flattened_image.append(reshaped_image[y][x])
        reshaped_letter_data.append(flattened_image)
        
        if save_images:
            save_letter(np.array(reshaped_image), count)

        print(f"reshaped {count+1} of {len(letter_data)} images...\r", end="")
    
    print("\nDone reshaping!")

    reshaped_letter_data = np.array(reshaped_letter_data)

    if save_file:
        np.save(f"reshaped_letter_data", reshaped_letter_data)

    return reshaped_letter_data


if __name__ == "__main__":
    with open('translated_letter_centroid_labels.npy', 'rb') as file:
        letter_data = np.load(file)

    reshaped_letter_data = reshape_letter_data(letter_data, save_images=False, save_file=False)
