import os
import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import load_model
from skimage.transform import resize
import os
import skimage.io
import numpy as np
import pandas as pd
import pickle

modelpath = "raw_data/models"
outDir = "raw_data/created_images"
extensions = [".jpg", ".jpeg"]
dataDir = "raw_data/abstract_ex2"
testDir = "raw_data/test_images"

class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

def resize_image(input_image):
    img = read_img(input_image)
    if np.array(img).ndim < 3 :
        img = np.atleast_3d(img)
    img = resize(img, (128,128))
    return img

def normalize_img(img):
    return img / 255.

def read_img(filePath):
    return skimage.io.imread(filePath, as_gray=False)

def resize_img(img, shape_resized):
    img_resized = resize(img, shape_resized,
                         anti_aliasing=True,
                         preserve_range=True)
    assert img_resized.shape == shape_resized
    return img_resized

def get_vgg19_model(shape_img, load_from_net = False):
    if load_from_net == True:
        model = VGG19(weights='imagenet', include_top=False,
                                             input_shape=shape_img)
    if load_from_net == False:
        model = load_model(os.path.join(modelpath,"vgg19_autoencoder.h5"))
    model.compile()
    return model

def get_art_info(style = "Abstract-Expressionism"):
    art_info = pd.read_csv("raw_data/wikiart_scraped.csv")
    art_info = art_info[art_info['Style']==style]
    return art_info

def apply_transformer(imgs, transformer):
    imgs_transform = [transformer(img) for img in imgs]
    return imgs_transform

def get_images_as_array(images, input_shape_model):
    images_as_array = np.array(images).reshape((-1,) + input_shape_model)
    return images_as_array

def get_flattened_array(images_as_array, output_shape_model):
    images_as_flattened_array = images_as_array.reshape((-1, np.prod(output_shape_model)))
    return images_as_flattened_array

def single_image_neighbours_info_as_dict(E_test_flatten, knn, art_info):
    related_images = []
    result = knn.kneighbors(E_test_flatten)
    img_ids = list(result[1][0])
    for i in img_ids:
        image_info = {}
        image_info['Image_filename'] = f"{art_info.iloc[i].name}.jpeg"
        image_info['Title'] = f"{art_info.iloc[i].Artwork}"
        image_info['Artist'] = f"{art_info.iloc[i].Artist}"
        related_images.append(image_info)
    return related_images

import matplotlib.pyplot as plt

def find_k_neighbours(image = "raw_data/test_images/26601.jpeg"):
    image = [read_img(image)]
    shape_img = image[0].shape
    output_shape_model = (4, 4, 512)
    # instantiate model
    model = get_vgg19_model(shape_img, load_from_net = False)
    transformer = ImageTransformer(shape_img)
    img_transformed = apply_transformer(image, transformer)
    X_test = get_images_as_array(img_transformed, shape_img)
    # transform into embeddings
    E_test = model.predict(X_test)
    E_test_flatten = get_flattened_array(E_test, output_shape_model)
    # process with knn
    knnfile = os.path.join(modelpath,"knnpickle_file")
    knn = pickle.load(open(knnfile, "rb"))
    art_info = get_art_info()
    ###
    result = single_image_neighbours_info_as_dict(E_test_flatten, knn, art_info)
    return result

if __name__ == "__main__":
    print(find_k_neighbours())
