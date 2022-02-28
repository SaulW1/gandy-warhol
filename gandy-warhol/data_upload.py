import pandas as pd
from google.cloud import storage
import urllib.request

bucket_name= "gandywarhol"

def get_data(style="Abstract-Expressionism"):
    df = pd.read_csv("raw_data/wikiart_scraped.csv")
    art_list=list(df[df['Style']==style]["Link"])
    return art_list

def upload_to_gcp():
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    art_list = get_data()
    for i in range(70,80):
        file = urllib.request.urlopen(art_list[i])
        blob = bucket.blob(f"data/{i}")
        blob.upload_from_string(file.read(),content_type="image/jpg")
        print(i)

if __name__  == "__main__":
    upload_to_gcp()
