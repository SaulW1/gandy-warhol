import pandas as pd
from google.cloud import storage
import urllib.request
import os
import requests


bucket_name= "gandywarhol"

def get_data(style="Abstract-Expressionism"):
    df = pd.read_csv("raw_data/wikiart_scraped.csv")
    art_list=list(df[df['Style']==style]["Link"])
    return art_list

def upload_to_gcp():
    # uploads files from links directly to Google Cloud bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    art_list = get_data()
    for i in range(70,80):
        file = urllib.request.urlopen(art_list[i])
        blob = bucket.blob(f"data/{i}")
        blob.upload_from_string(file.read(),content_type="image/jpg")
        print(i)

def download_files():
    art_list = get_data(style="Abstract-Expressionism")
    errors = []
    for i in range(len(art_list)):
        if not os.path.exists('raw_data'):
            os.makedirs('raw_data')  # create folder if it does not exist
        filename = f'{i}.jpeg'  # be careful with file names
        file_path = os.path.join('raw_data', filename)

        r = requests.get(art_list[i], stream=True)
        if r.ok:
            print(f"saving {i} to", os.path.abspath(file_path))
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
        else:  # HTTP status code 4XX/5XX
            errors.append(art_list[i])
    print(errors)


if __name__  == "__main__":
    download_files()
