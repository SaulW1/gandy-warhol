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

modelpath = "models"
outDir = "raw_data/created_images"
extensions = [".jpg", ".jpeg"]
dataDir = "/content/drive/MyDrive/lewagon_gandy/abstract_ex2"

def __init__(self, modelName, info):
    self.modelName = modelName
    self.info = info
    self.autoencoder = None
    self.encoder = None
    self.decoder = None

def predict(self, X):
    return self.encoder.predict(X)

def load_models(self, loss="binary_crossentropy", optimizer="adam", path = modelpath):
    print("Loading models...")
    self.autoencoder = tf.keras.models.load_model(self.info[f"{path}/autoencoderFile"])
    self.encoder = tf.keras.models.load_model(self.info[f"{path}/encoderFile"])
    self.decoder = tf.keras.models.load_model(self.info[f"{path}/decoderFile"])
    self.autoencoder.compile(optimizer=optimizer, loss=loss)
    self.encoder.compile(optimizer=optimizer, loss=loss)
    self.decoder.compile(optimizer=optimizer, loss=loss)
    return self

def compile(self, loss="binary_crossentropy", optimizer="adam"):
    self.autoencoder.compile(optimizer=optimizer, loss=loss)
    return self

def save_models(self, path = modelpath):
    print("Saving models...")
    self.autoencoder.save(self.info[f"{path}/autoencoderFile"])
    self.encoder.save(self.info[f"{path}/encoderFile"])
    self.decoder.save(self.info[f"{path}/decoderFile"])
    return self

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
    if outFile is None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
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



if __name__ == '__main__':
    all_images = read_imgs_dir(dataDir, extensions)
