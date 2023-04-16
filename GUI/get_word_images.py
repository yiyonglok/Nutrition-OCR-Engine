from PIL import Image, ImageOps
import numpy as np
import os


def get_word_image_paths(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file or '.jpg' in file:
                files.append(os.path.join(r, file))
    return files


def get_word_images(nutrition_label_image, heatmap_image):
    with Image.open(nutrition_label_image) as img:
        img = ImageOps.grayscale(img)
        img_pixel_data = img.load()
        width, height = img.size

    with Image.open(heatmap_image) as heatmap:
        heatmap = ImageOps.grayscale(heatmap)
        img_pixel_data_heatmap = heatmap.load()

    # get lines
    lines = []
    count = 0
    for y in range(height):
        for x in range(width):
            if img_pixel_data[x, y] == 0:
                count += 1
        if count >= 150:
            lines.append(y)
        count = 0
    # get lowest lines
    lines_ = []
    for i in range(len(lines) - 1):
        if lines[i + 1] - lines[i] >= 25:
            lines_.append(lines[i])
    # draw lines
    for line in lines_:
        for x in range(width):
            img_pixel_data[x, line] = 0
    if img.mode != 'RGB':
        image = img.convert('RGB')
    img.save(f"lines.jpg")

    # get pieces from heatmap
    count = 0
    prev = lines_[0]
    for i in range(1, len(lines_)):
        curr = lines_[i]
        pieces = []
        for j in range(prev, curr):
            temp = []
            for x in range(width):
                temp.append(img_pixel_data_heatmap[x, j])
            pieces.append(temp)

        # create image from piece
        # image = Image.fromarray(np.array(pieces))
        image = Image.fromarray(np.array(pieces).astype(np.uint8))

        img_pixel_data_piece = image.load()
        width, height = image.size
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(f"pieces_heatmap/piece_{count}.jpg")

        # boost contrast of heatmap piece
        # need to reload image data to create new images from same array
        image = Image.fromarray(np.array(pieces).astype(np.int8))
        img_pixel_data_piece = image.load()
        for i_index in range(height):
            for j_index in range(width):
                if img_pixel_data_piece[j_index, i_index] > 0:
                    img_pixel_data_piece[j_index, i_index] = 255
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(f"pieces_heatmap_contrast/piece_{count}.jpg")

        # find word intervals in piece
        mid = height // 2
        consecutive = 0
        x_indices = []
        for i in range(width):
            if img_pixel_data_piece[i, mid] == 255:
                if consecutive == 0:
                    x_index = i
                consecutive += 1
            else:
                if consecutive >= 60:
                    x_indices.append((x_index, i))
                consecutive = 0
        # print(x_indices)

        # create word images
        for word_count, index in enumerate(x_indices):
            word = []
            for j in range(prev, curr):
                temp = []
                for x in range(index[0], index[1] + 1):
                    temp.append(img_pixel_data[x, j])
                word.append(temp)
            #word_image = Image.fromarray(np.array(word))
            word_image = Image.fromarray(np.array(word).astype(np.uint8))

            if word_image.mode != 'RGB':
                word_image = word_image.convert('RGB')
            word_image.save(f"words/word_{count}_{word_count}.jpg")

        # iterate counter and previous line
        count += 1
        prev = curr


if __name__ == "__main__":
    get_word_images(nutrition_label_image="label.jpg", heatmap_image="heatmap.jpg")