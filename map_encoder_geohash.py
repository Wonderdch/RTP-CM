import pygeohash as geohash
import pandas as pd


def geohashKey_to_token(geohash_key):
    token_string = geohash_key[2:8]
    return token_string


def create_location_vocab(city):
    # 让相近的 key 生成的 token 靠近
    data = pd.read_csv(f"./raw_data/{city}_poi_mapping.csv")
    token_list = []
    for index, row in data.iterrows():
        geohash_key = geohash.encode(row["latitude"], row["longitude"], 8)
        token_list.append(geohashKey_to_token(geohash_key))
    token_list.sort()
    vocab = {}
    index = 0
    for token in token_list:
        if token in vocab:
            continue
        else:
            vocab[token] = index
            index += 1
    return vocab


def get_location_vector(latitude: float, longitude: float):
    geohash_key = geohash.encode(latitude, longitude, 8)
    return location_vocab[geohashKey_to_token(geohash_key)]


location_vocab = create_location_vocab(city="PHO")
