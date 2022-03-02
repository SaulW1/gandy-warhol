from gandywarhol.params import *
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def get_data(image_path = 'raw_data/abstract_ex'):
    all_images = []
    for filename in os.listdir(image_path):
        if filename.endswith('.jpeg'):
            path = os.path.join(image_path, filename)
            image = Image.open(path).resize((128, 128), Image.ANTIALIAS)
            all_images.append(np.asarray(image))
    all_images = np.array(all_images)
    all_images.reshape(-1, 128, 128, 3)
    return all_images

def train_and_test(all_images):
    X_train, X_test = train_test_split(all_images)
    return X_train, X_test

def build_encoder(latent_dimension):
    '''returns an encoder model, of output_shape equals to latent_dimension'''
    encoder = Sequential()

    encoder.add(Conv2D(8, (2,2), input_shape=(28, 28, 1), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(16, (2, 2), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(32, (2, 2), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Flatten())
    encoder.add(Dense(latent_dimension, activation='tanh'))

    return encoder


if __name__ == '__main__':
    all_images = get_data()
    X_train, X_test = train_and_test(all_images)
    print(X_train.shape)
    print(X_test.shape)
