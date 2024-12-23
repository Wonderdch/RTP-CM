# https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system?redirectedfrom=MSDN

import math

import pandas as pd
from torchtext.vocab import build_vocab_from_iterator

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180


# Clips a number to the specified minimum and maximum values.
def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)


# Determines the map width and height (in pixels) at a specified level of detail.
def map_size(levelOfDetail):
    return 256 << levelOfDetail


# Converts a point from latitude/longitude into pixel XY coordinates at a specified level of detail.
def latlong2pixel_xy(latitude, longitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY


# Converts pixel XY coordinates into tile XY coordinates of the tile containing the specified pixel.
def pixel_xy2tile_xy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY


# Converts tile XY coordinates into a QuadKey at a specified level of detail.
def tile_xy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return ''.join(quadKey)


# Converts a point from latitude/longitude into a QuadKey at a specified level of detail.
def latlon2quadkey(lat, lon, level):
    pixelX, pixelY = latlong2pixel_xy(lat, lon, level)
    tileX, tileY = pixel_xy2tile_xy(pixelX, pixelY)
    return tile_xy2quadkey(tileX, tileY, level)


def yield_tokens(data):
    for index, row in data.iterrows():
        quadkey = latlon2quadkey(row['latitude'], row['longitude'], 17)
        yield [quadkey[9:17]]


def create_location_vocab():
    data = pd.read_csv(f"./raw_data/PHO_poi_mapping.csv")
    return build_vocab_from_iterator(yield_tokens(data))


def get_location_vector(latitude: float, longitude: float):
    quadkey = latlon2quadkey(latitude, longitude, 17)
    return location_vocab_geohash([quadkey[9:17]])


location_vocab_geohash = create_location_vocab()
