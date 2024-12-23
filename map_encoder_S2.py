import pandas as pd
from torchtext.vocab import build_vocab_from_iterator

import S2.sphere


def yield_tokens(data):
    for index, row in data.iterrows():
        cell = S2.sphere.CellId().from_lat_lng(S2.sphere.LatLng.from_degrees(row['latitude'], row['longitude']))
        s2_key = cell.parent(16).to_token()
        yield [s2_key[4:9]]
        # s2_key = s2_key[4:9]
        # token = ' '.join([''.join(x) for x in ngrams(s2_key, 3)])
        # yield token.split(' ')


def create_location_vocab():
    data = pd.read_csv(f"./raw_data/PHO_poi_mapping.csv")
    return build_vocab_from_iterator(yield_tokens(data))


def get_location_vector(latitude: float, longitude: float):
    cell = S2.sphere.CellId().from_lat_lng(S2.sphere.LatLng.from_degrees(latitude, longitude))
    s2_key = cell.parent(16).to_token()
    return location_vocab([s2_key[4:9]])


location_vocab = create_location_vocab()
