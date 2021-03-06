import tensorflow as tf
import random
import numpy as np
from gandywarhol.params import *


class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(
        self, X, batch_size=32, input_size=input_size, shuffle=True, resizer=False
    ):

        self.X = X.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.resizer = resizer

        self.n = len(self.X)

    def on_epoch_end(self):
        if self.shuffle:
            print("Reshuffling data...")
            random.shuffle(self.X)

    def img_loader(self, path):
        return np.array(tf.keras.utils.load_img(path))

    def resize(self, img):
        return tf.image.resize(img, self.input_size)

    def __getitem__(self, index):
        temp_list = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            img = self.img_loader(self.X[i])
            if self.resizer == True:
                img = self.resize(img)
            temp_list.append(img)
        return np.stack(temp_list)

    def __len__(self):
        return self.n // self.batch_size
