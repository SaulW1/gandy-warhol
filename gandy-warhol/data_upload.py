import pandas as pd
import numpy as np
from google.cloud import storage
import urllib.request
import os
import requests
from params import *
import tensorflow.image
import tensorflow.keras.utils


def get_data(style="Abstract-Expressionism"):
    df = pd.read_csv("raw_data/wikiart_scraped.csv")
    art_list=list(df[df['Style']==style]["Link"])
    return art_list

def download_files(directory = 'raw_data/abstract_ex'):
    art_list = get_data(style="Abstract-Expressionism")
    errors = []
    if not os.path.exists(directory):
        os.makedirs(directory)  # create folder if it does not exist
    for i in range(len(art_list)):
        filename = f'{i}.jpeg'  # be careful with file names
        file_path = os.path.join(directory, filename)
        try:
            r = requests.get(art_list[i], stream=True)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
                img = tensorflow.keras.utils.load_img(file_path)
                if np.array(img).ndim < 3:
                    img = np.atleast_3d(img)
                img_resized = tensorflow.image.resize(img, input_size)
                tensorflow.keras.utils.save_img(file_path, img_resized)
                print(f"saved {i} to", os.path.abspath(file_path))
        except:  # Handle issues
            errors.append(art_list[i])
            print(f"Error: {i}")
    print(f"Errors: {errors}")


if __name__  == "__main__":
    download_files()
