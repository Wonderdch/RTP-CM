import h3
import pandas as pd


def yield_tokens(data):
    for index, row in data.iterrows():
        h3_key = h3.latlng_to_cell(row['latitude'], row['longitude'], 10)
        yield h3Key_to_token(h3_key)


# https://observablehq.com/@nrabinowitz/h3-index-bit-layout?collection=@nrabinowitz/h3
def h3Key_to_token(h3_key):
    b64 = f'{int(h3_key, base=16):08b}'
    valid_string = b64[15:15 + 30]

    sub_strings = []
    for i in range(0, len(valid_string), 3):
        sub_strings.append(valid_string[i:i + 3])

    token_int = []
    for sub_string in sub_strings:
        token_int.append(int(sub_string, base=2))

    string_list = [str(n) for n in token_int]
    token_string = "".join(string_list)
    return token_string


def create_location_vocab(city):
    data = pd.read_csv(f"./raw_data/{city}_poi_mapping.csv")
    token_list = []
    for index, row in data.iterrows():
        h3_key = h3.latlng_to_cell(row['latitude'], row['longitude'], 10)
        token_list.append(h3Key_to_token(h3_key))
    token_list.sort()
    h3_vocab = {}
    index = 0
    for token in token_list:
        if token in h3_vocab:
            continue
        else:
            h3_vocab[token] = index
            index += 1
    return h3_vocab


def get_location_vector(latitude: float, longitude: float):
    h3_key = h3.latlng_to_cell(latitude, longitude, 10)
    return location_vocab[h3Key_to_token(h3_key)]


location_vocab = create_location_vocab(city="PHO")
