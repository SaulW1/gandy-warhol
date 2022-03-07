"""tests for data_upload.py functions"""

from gandywarhol.data_upload import *
import os
import tensorflow.image
import tensorflow.keras.utils

def test_data_upload():
    assert type(get_data(path = "raw_data/wikiart_scraped.csv")) == list

def test_processed_images(images_path = "raw_data/abstract_ex2"):
    images_path = images_path
    errors = []
    for filename in os.listdir(images_path):
        if filename.endswith('.jpeg'):
            img = tensorflow.keras.utils.load_img(os.path.join(images_path, filename))
            if np.array(img).shape != (128,128,3):
                errors.append(filename)
    print(errors)


if __name__ == '__main__':
    test_processed_images()
