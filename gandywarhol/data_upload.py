import pandas as pd
import numpy as np
import os
import requests
from gandywarhol.params import *
import tensorflow.image
import tensorflow.keras.utils


def get_data(style="Abstract-Expressionism", path = "raw_data/wikiart_scraped.csv"):
    df = pd.read_csv(path)
    art_list = df[df['Style']==style]
    return art_list

def download_files(directory = 'raw_data/abstract_ex2', style ="Abstract-Expressionism", howmany = 5):
    art_list = get_data(style=style).reset_index()
    art_list_links = list(art_list["Link"])
    errors = []
    if not os.path.exists(directory):
        os.makedirs(directory)  # create folder if it does not exist
    for i in range(howmany):
        indexno = art_list.iloc[i]['index']
        filename = f'{indexno}.jpeg'  # be careful with file names
        file_path = os.path.join(directory, filename)
        try:
            r = requests.get(art_list_links[i], stream=True)
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
    download_files(directory = "raw_data/imp", style="Impressionism")
