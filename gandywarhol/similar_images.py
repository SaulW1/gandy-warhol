import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from skimage.transform import resize
import os
import skimage.io
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

modelpath = "raw_data/models"
outDir = "raw_data/created_images"
extensions = [".jpg", ".jpeg"]
dataDir = "raw_data/abstract_ex2"
testDir = "raw_data/test_images"

# def __init__(self, modelName, info):
#     self.modelName = modelName
#     self.info = info
#     self.autoencoder = None
#     self.encoder = None
#     self.decoder = None

# def predict(self, X):
#     return self.encoder.predict(X)


class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

# def load_models(self, loss="binary_crossentropy", optimizer="adam", path = modelpath):
#     print("Loading models...")
#     self.autoencoder = tf.keras.models.load_model(self.info[f"{path}/autoencoderFile"])
#     self.encoder = tf.keras.models.load_model(self.info[f"{path}/encoderFile"])
#     self.decoder = tf.keras.models.load_model(self.info[f"{path}/decoderFile"])
#     self.autoencoder.compile(optimizer=optimizer, loss=loss)
#     self.encoder.compile(optimizer=optimizer, loss=loss)
#     self.decoder.compile(optimizer=optimizer, loss=loss)
#     return self

# def compile(self, loss="binary_crossentropy", optimizer="adam"):
#     self.autoencoder.compile(optimizer=optimizer, loss=loss)
#     return self

# def save_models(self, path = modelpath):
#     print("Saving models...")
#     self.autoencoder.save(self.info[f"{path}/autoencoderFile"])
#     self.encoder.save(self.info[f"{path}/encoderFile"])
#     self.decoder.save(self.info[f"{path}/decoderFile"])
#     return self

def read_img(filePath):
    return skimage.io.imread(filePath, as_gray=False)

def read_imgs_dir(dirPath, extensions):
    args = [os.path.join(dirPath, filename)
            for filename in os.listdir(dirPath)
            if any(filename.lower().endswith(ext) for ext in extensions)]
    args_sorted = sorted(args)
    imgs = [read_img(arg) for arg in args_sorted]
    return imgs

def save_img(filePath, img):
    skimage.io.imsave(filePath, img)

def plot_img(img, range=[0, 255]):
    plt.imshow(img, vmin=range[0], vmax=range[1])
    plt.xlabel("xpixels")
    plt.ylabel("ypixels")
    plt.tight_layout()
    plt.show()
    plt.close()

def apply_transformer(imgs, transformer):
    imgs_transform = [transformer(img) for img in imgs]
    return imgs_transform

def normalize_img(img):
    return img / 255.

def resize_img(img, shape_resized):
    img_resized = resize(img, shape_resized,
                         anti_aliasing=True,
                         preserve_range=True)
    assert img_resized.shape == shape_resized
    return img_resized

def flatten_img(img):
    return img.flatten("C")

def split(fracs, N, seed):
    fracs = [round(frac, 2) for frac in fracs]
    if sum(fracs) != 1.00:
        raise Exception("fracs do not sum to one!")

    # Shuffle ordered indices
    indices = list(range(N))
    random.Random(seed).shuffle(indices)
    indices = np.array(indices, dtype=int)

    # Get numbers per group
    n_fracs = []
    for i in range(len(fracs) - 1):
        n_fracs.append(int(max(fracs[i] * N, 0)))
    n_fracs.append(int(max(N - sum(n_fracs), 0)))

    if sum(n_fracs) != N:
        raise Exception("n_fracs do not sum to N!")

    # Sample indices
    n_selected = 0
    indices_fracs = []
    for n_frac in n_fracs:
        indices_frac = indices[n_selected:n_selected + n_frac]
        indices_fracs.append(indices_frac)
        n_selected += n_frac

    # Check no intersections
    for a, indices_frac_A in enumerate(indices_fracs):
        for b, indices_frac_B in enumerate(indices_fracs):
            if a == b:
                continue
            if is_intersect(indices_frac_A, indices_frac_B):
                raise Exception("there are intersections!")

    return indices_fracs

def is_intersect(arr1, arr2):
    n_intersect = len(np.intersect1d(arr1, arr2))
    if n_intersect == 0:
        return False
    return True

def get_vgg19_model(load_from_net = False):
    if load_from_net == True:
        model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                             input_shape=shape_img)
    if load_from_net == False:
        model = tf.keras.models.load_model("models/vgg19_autoencoder.h5")
    model.compile()
    return model

def get_shapes(model):
    shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
    input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
    return shape_img_resize, input_shape_model, output_shape_model

def get_art_info(style = "Abstract-Expressionism"):
    art_info = pd.read_csv("raw_data/wikiart_scraped.csv")
    art_info = art_info[art_info['Style']==style]
    return art_info

def get_images_as_flattened_array(images, shape_model):
    images_as_array = np.array(images).reshape((-1,) + shape_model)
    images_as_flattened_array = np.array(images).reshape((-1,) + shape_model)
    return images_as_flattened_array

def get_embeddings_with_knn(train_images):
    knn = NearestNeighbors(n_neighbors=4, metric="cosine")
    knn.fit(train_images)
    return knn

def get_neighbours_info_as_dict(X_test, knn, test_images, all_images, art_info):
    knn_info = []
    for i, emb_flatten in enumerate(X_test):
        _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
        img_query = test_images[i] # query image
        imgs_retrieval = [all_images[idx] for idx in indices.flatten()] # retrieval images
        img_ids = [idx for idx in indices.flatten()]
        pic_info = dict_query_retrieval(img_query, imgs_retrieval, img_ids, art_info)
        knn_info.append(pic_info)

def dict_query_retrieval(img_query, imgs_retrieval, img_ids, art_info="art_info",):
    info_list = [img_query]
    for i, img in enumerate(imgs_retrieval):
        imdict = {}
        imdict['Image'] = img
        imdict['Title'] = f"{art_info.iloc[img_ids[i]].Artwork}"
        imdict['Artist'] = f"{art_info.iloc[img_ids[i]].Artist}"
        info_list.append(imdict)
    return info_list

def get_neighbours_info_as_image(X_test, knn, test_images, all_images, art_info):
    for i, emb_flatten in enumerate(X_test):
        art_info = {}
        _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
        img_query = test_images[i] # query image
        imgs_retrieval = [all_images[idx] for idx in indices.flatten()] # retrieval images
        img_ids = [idx for idx in indices.flatten()]
        plot_query_retrieval(img_query, imgs_retrieval, img_ids, art_info)

def plot_query_retrieval(img_query, imgs_retrieval, img_ids, outFile, art_info="art_info",):
    n_retrieval = len(imgs_retrieval)
    fig = plt.figure(figsize=(2*n_retrieval, 4))
    fig.suptitle(f"Similar images")
    # Plot query image
    ax = plt.subplot(2, n_retrieval, 0 + 1)
    plt.imshow(img_query)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)  # increase border thickness
        ax.spines[axis].set_color('black')  # set to black
    ax.set_title("query",  fontsize=10)  # set subplot title
    count = 0
    # Plot retrieval images
    for i, img in enumerate(imgs_retrieval):
        ax = plt.subplot(2, n_retrieval, n_retrieval + i + 1)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_title = f"{art_info.iloc[img_ids[count]].Artwork}"
        img_artist = f"{art_info.iloc[img_ids[count]].Artist}"
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)  # set border thickness
            ax.spines[axis].set_color('black')  # set to black
        ax.set_title(f"{img_title} by {img_artist}", fontsize=10)  # set subplot title
        count+=1
    result = plt.show()
    return result

def full_sequence(train_models = True, outputs = "dict"):
    # get images
    all_images = read_imgs_dir(dataDir, extensions)
    test_images = read_imgs_dir(testDir, extensions)
    # instantiate model
    model = get_vgg19_model(loading = False)
    # establish img sizes and transform
    shape_img_resize, input_shape_model, output_shape_model = get_shapes(model)
    transformer = ImageTransformer(shape_img_resize)
    imgs_train_transformed = apply_transformer(all_images, transformer)
    imgs_test_transformed = apply_transformer(test_images, transformer)
    X_train = get_images_as_flattened_array(imgs_train_transformed, output_shape_model)
    X_test = get_images_as_flattened_array(imgs_test_transformed, input_shape_model)
    # process with knn
    if train_models == True:
        knn = get_embeddings_with_knn(X_train) #replace this with importing pickle file
    else:
        knn = pickle.load(open(os.path.join(modelpath,"knnpickle.pkl")))
    knn.fit(X_test)
    # get art metadata
    art_info = get_art_info()
    # find neighbours and return result
    if outputs == "dict":
        result = get_neighbours_info_as_dict(X_test, knn, test_images, all_images, art_info)
    elif outputs == "plot":
        result = get_neighbours_info_as_image(X_test, knn, test_images, all_images, art_info)
    return result


if __name__ == '__main__':
    all_images = read_imgs_dir(dataDir, extensions)
    test_images = read_imgs_dir(testDir, extensions)
    print(type(all_images))
    print(len(test_images))
