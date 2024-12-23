import math
import os
import pickle
import random
from datetime import timedelta

import h3
import numpy as np
import pandas as pd

import S2.sphere
from map_encoder_tile import latlon2quadkey

# constants
min_seq_len = 3
min_seq_num = 3
min_short_term_len = 5
min_long_term_count = 2
pre_seq_window = 7
random_seed = 2021


# split sequence
def generate_sequence(input_data, min_seq_len, min_seq_num):
    """Split and filter action sequences for each user

    Args:
        input_data (DataFrame): raw data read from csv file
        min_seq_len (int): minimum length for a sequence to be considered valid
        min_seq_num (int): minimum no. sequences for a user to be considered valid

    Returns:
        total_sequences_dict ({user_id: [[visit_id]]}): daily action sequences for each user
        total_sequences_meta ([(user)[(timestamp(int),seq_len(int))]]) : date and length for each sequnece
    """

    def _remove_consecutive_visit(visit_record):
        """remove duplicated consecutive visits in a sequence

        Args:
            visit_record (DataFrame): raw sequences

        Returns:
            clean_sequence (list): sequences with no duplicated consecutive visits
        """
        clean_sequence = []
        for index, _ in visit_record.iterrows():
            clean_sequence.append(index)
        return clean_sequence

    total_sequences_dict = {}  # records visit id in each sequence
    total_sequences_meta = []  # records sequence date and length

    seq_count = 0  # for statistics only

    input_data['Local_sg_time'] = pd.to_datetime(input_data['Local_Time_True'])

    for user in input_data['UserId'].unique():  # process sequences for each user
        user_visits = input_data[input_data['UserId'] == user]
        user_sequences, user_sequences_meta = [], []
        unique_date_group = user_visits.groupby([user_visits['Local_sg_time'].dt.date])
        for date in unique_date_group.groups:  # process sequences on each day
            single_date_visit = unique_date_group.get_group(date)
            single_sequence = _remove_consecutive_visit(single_date_visit)
            if len(single_sequence) >= min_seq_len:  # filter sequences too short
                user_sequences.append(single_sequence)
                user_sequences_meta.append((date, len(single_sequence)))
                seq_count += 1
        if len(user_sequences) >= min_seq_num:  # filter users with too few visits
            total_sequences_dict[user] = np.array(user_sequences, dtype=object)
            total_sequences_meta.append(user_sequences_meta)
    print(f"Generated {seq_count} sequences in total for {len(total_sequences_dict.keys())} users")
    return total_sequences_dict, total_sequences_meta


# generate sequences of different features
def _reIndex_3d_list(input_list):
    """Reindex the elements in sequences

    Args:
        input_list (nd_array: [(all users)[(user list)[(seq list)]]]): a 3d list containing all sequences fofr all users

    Returns:
        reIndexed_list (3d list): reindexed list
        index_map ([id]): each element is an original id, the index of an element is the new index the id
    """

    def _flatten_3d_list(input_list):
        """flattern a 3d list to 1d

        Args:
            input_list (nd_array: [(all users)[(user list)[(seq list)]]]): a 3d list containing all sequences fofr all users

        Returns:
            1d-list: flattened list
        """
        twoD_lists = input_list.flatten()
        return np.hstack([np.hstack(twoD_list) for twoD_list in twoD_lists])

    def _old_id_to_new(mapping, old_id):
        """convert old_id to new index by mapping given

        Args:
            mapping ([id]): each element is an original id, the index of an element is the new index the id
            old_id (Any): the original token/id in the list

        Returns:
            int: new index of the token
        """
        return np.where(mapping == old_id)[0].flat[0]

    flat_list = _flatten_3d_list(input_list)  # make 3d list 1d
    index_map = np.unique(flat_list)  # get
    reIndexed_list = []
    for user_seq in input_list:  # seq list for each user
        reIndexed_user_list = []
        for seq in user_seq:  # each seq
            reIndexed_user_list.append([_old_id_to_new(index_map, poi) for poi in seq])
        reIndexed_list.append(reIndexed_user_list)
    reIndexed_list = np.array(reIndexed_list, dtype=object)

    return reIndexed_list, index_map


def generate_POI_sequences(input_data, visit_sequence_dict):
    """generate location transition sequences

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_POI_sequences (nd_array: [[[POI_index]]]): daily location transition sequences for each user
        POI_reIndex_mapping ([POI_id]): index is the new POI index and element is the original POI id
    """
    POI_sequences = []

    for user in visit_sequence_dict:
        user_POI_sequences = []
        for seq in visit_sequence_dict[user]:
            single_POI_sequence = []
            for visit in seq:
                single_POI_sequence.append(input_data['VenueId'][visit])
            user_POI_sequences.append(single_POI_sequence)
        POI_sequences.append(user_POI_sequences)
    reIndexed_POI_sequences, POI_reIndex_mapping = _reIndex_3d_list(np.array(POI_sequences, dtype=object))
    return reIndexed_POI_sequences, POI_reIndex_mapping


def generate_category_sequences(input_data, visit_sequence_dict):
    """generate category transition sequences

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_cat_sequences (nd_array: [[[cat_index]]]): daily category transition sequences for each user
        cat_reIndex_mapping ([cat_name]): index is the new category index and element is the original category name
    """
    cat_sequences = []
    for user in visit_sequence_dict:
        user_cat_sequences = []
        for seq in visit_sequence_dict[user]:
            single_cat_sequence = []
            for visit in seq:
                # 用的是大类，一共 10 个类别
                # 如果是 Category 的话，有 239 个类别
                single_cat_sequence.append(input_data['L1_Category'][visit])
            user_cat_sequences.append(single_cat_sequence)
        cat_sequences.append(user_cat_sequences)
    reIndexed_cat_sequences, cat_reIndex_mapping = _reIndex_3d_list(np.array(cat_sequences, dtype=object))
    return reIndexed_cat_sequences, cat_reIndex_mapping


def generate_user_sequences(input_data, visit_sequence_dict):
    """generate time (in hour) transition sequences

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_user_sequences (nd_array: [[[user_index]]]): daily user sequences (same for each sequence)
        user_reIndex_mapping ([user_id]): index is the new user index and element is the original user id
    """
    all_user_sequences = []
    for user in visit_sequence_dict:
        user_sequences = []
        for seq in visit_sequence_dict[user]:
            single_user_sequence = [user] * len(seq)
            user_sequences.append(single_user_sequence)
        all_user_sequences.append(user_sequences)
    reIndexed_user_sequences, user_reIndex_mapping = _reIndex_3d_list(np.array(all_user_sequences, dtype=object))
    return reIndexed_user_sequences, user_reIndex_mapping


def generate_hour_sequences(input_data, visit_sequence_dict):
    """generate time (in hour) transition sequences

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_hour_sequences (nd_array: [[[time_index]]]): daily hour transition sequences for each user
        hour_reIndex_mapping ([hour]): index is the new hour index and element is the original hour
    """
    input_data["hour"] = pd.to_datetime(input_data['Local_Time_True']).dt.hour  # add hour column in raw data

    hour_sequences = []
    for user in visit_sequence_dict:
        user_hour_sequences = []
        for seq in visit_sequence_dict[user]:
            single_hour_sequence = []
            for visit in seq:
                single_hour_sequence.append(input_data['hour'][visit])
            user_hour_sequences.append(single_hour_sequence)
        hour_sequences.append(user_hour_sequences)
    reIndexed_hour_sequences, hour_reIndex_mapping = _reIndex_3d_list(np.array(hour_sequences, dtype=object))
    return reIndexed_hour_sequences, hour_reIndex_mapping


def generate_day_sequences(input_data, visit_sequence_dict):
    """generate weekday/weekend tag for each sequence

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_day_sequences (nd_array: [[[day_index]]]): daily weekday/weekend sequences (same for each sequence)
        day_reIndex_mapping ([weekday/weekend]): index is the new day index and element is the original weekday(False)/weekend(True) tag
    """
    input_data["is_weekend"] = pd.to_datetime(
        input_data['Local_Time_True']).dt.dayofweek > 4  # add hour column in raw data

    day_sequences = []
    for user in visit_sequence_dict:
        user_day_sequences = []
        for seq in visit_sequence_dict[user]:
            single_day_sequence = []
            for visit in seq:
                single_day_sequence.append(input_data['is_weekend'][visit])
            user_day_sequences.append(single_day_sequence)
        day_sequences.append(user_day_sequences)
    reIndexed_day_sequences, day_reIndex_mapping = _reIndex_3d_list(np.array(day_sequences, dtype=object))
    return reIndexed_day_sequences, day_reIndex_mapping


def generate_dist_matrix_sequences(input_data, visit_sequence_dict):
    """generate distance matrix for sequences
        e.g., for a sequence [1,2,3], the dist matrix sequences would be:
            [[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]]

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        dist_matrices (nd_array: [[dist_matrix]]): dist matrix for each daily sequences for each user
    """

    def _get_distance(pos1, pos2):
        """Calculate the between two  positions

        Args:
            pos1 ((lat, lon)): coordinates for the position 1
            pos2 ((lat, lon)): coordinates for the position 2

        Returns:
            h_dist (float): distances between two positions
        """
        lat1, lon1 = pos1
        lat2, lon2 = pos2

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(math.radians(dlat / 2)) ** 2 + math.cos(math.radians(lat1)) * math.cos(
            math.radians(lat2)) * math.sin(math.radians(dlon / 2)) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        h_dist = c * r

        return h_dist

    def _generate_dist_matrix(seq, input_data):
        """Generate a distance matrix for one sequence

        Args:
            seq ([visit_id]): one visit sequence
            input_data (DataFrame): raw check-in data

        Returns:
            dist_matrix ([[d11,d12,...],[d21,d22,...],...]): a matrix show distance between each pair of POI in the sequence
        """
        return [[_get_distance((input_data['Latitude'][x], input_data['Longitude'][x]),
                               (input_data['Latitude'][y], input_data['Longitude'][y])) \
                 for x in seq] for y in seq]

    dist_matrices = []
    for user in visit_sequence_dict:
        user_dist_matrices = []
        for seq in visit_sequence_dict[user]:  # generate dist matrix for each seq
            dist_matrix = _generate_dist_matrix(seq, input_data)
            user_dist_matrices.append(dist_matrix)
        dist_matrices.append(user_dist_matrices)
    return np.array(dist_matrices, dtype=object)


# generate (short term + long term) feed data

def filter_long_short_term_sequences(total_sequences_meta, min_short_term_len, pre_seq_window, min_long_term_count):
    """filter valid long+short-term sequences for generation of input data
        criteria: 1. the feed data is composed of multiple long-term sequences and one short-term sequence;
                  2. the short term sequence length >= min_short_term_len(5)
                  3. the long term sequences a sequences 7 days before the short term sequence
                  4. the number of long term sequences must >= min_long_term_count(2)

    Args:
        total_sequences_meta ([(user)[(timestamp(int),seq_len(int))]]): date and length for each sequnece
        min_short_term_len (int): minimum visits in a short-term sequence
        pre_seq_window (int): number of days to look for long-term sequences
        min_long_term_count (int): minimum number of long-term sequences to make the long-short sequences valid

    Returns:
        valid_input_index ([(all users)[(each user)[(valid sequences)seq_index]]]): valid long+short term sequneces for each user
    """

    valid_input_index = []  # filtered long+short term data
    valid_user_count, valid_input_count = 0, 0,  # for statistics purpose

    for _, user_sequences in enumerate(total_sequences_meta):  # for each user
        user_valid_input_index = []
        # print(user_sequences)
        for seq_index, seq in enumerate(user_sequences):  # for each sequence
            # print(seq)
            # print(seq_index)
            seq_time, seq_len = seq[0], seq[1]
            if seq_len >= min_short_term_len:  # valid short-term sequence
                start_time, end_time = seq_time - timedelta(days=pre_seq_window), seq_time
                long_term_seqs = [(index, seq) for index, seq in enumerate(user_sequences[:seq_index]) if
                                  start_time <= seq[0] <= end_time]
                if len(long_term_seqs) >= min_long_term_count:  # valid long-short term sequence
                    user_valid_input_index.append([seq[0] for seq in long_term_seqs] + [seq_index])
                    valid_input_count += 1
        valid_input_index.append(user_valid_input_index)
        valid_user_count += 1 if len(user_valid_input_index) > 0 else 0

    print(f"Filtered {valid_input_count} valid input long+short term sequences for {valid_user_count} users.")
    return valid_input_index


def generate_input_samples(feature_sequences, valid_input_index):
    """turn a feature sequence into a input long+short term data to be fed into model

    Args:
        feature_sequences (nd_array: [[[id]]]): daily transition sequences for each user for certain feature
        valid_input_index ([(all users)[(each user)[(valid sequences)seq_index]]]): valid long+short term sequneces for each user

    Return:
        input_samples ([(input sample)[(sequences)feature_id]]]): valid long+short term feature sequences for each user
    """
    input_samples = []
    for user_index, user_sequences in enumerate(valid_input_index):
        if len(user_sequences) != 0:
            for seq in user_sequences:
                feature_sequence = [feature_sequences[user_index][index] for index in seq]
                input_samples.append(feature_sequence)
    return input_samples


def split_train_test(input_samples):
    """split a input sequence into training, validation and testing sequences
        criteria: train-80%, validation-10%, test-10%

    Args:
        input_samples (3d array: [(each sample)[(valid sequences)feature_id]]): valid long+short term feature sequences for each user

    Returns:
        all_training_samples: 80% of samples for training
        all_validation_samples: 10% of samples for validation
        all_testing_samples: 10% of samples for testing
        all_training_validation_samples: 90% of samples for final training after validation
    """
    random.Random(random_seed).shuffle(input_samples)
    N = len(input_samples)
    train_valid_boundary = int(0.8 * N)
    valid_test_boundary = int(0.9 * N)
    all_training_samples = input_samples[:train_valid_boundary]
    all_validation_samples = input_samples[train_valid_boundary:valid_test_boundary]
    all_testing_samples = input_samples[valid_test_boundary:]
    all_training_validation_samples = input_samples[:valid_test_boundary]

    return all_training_samples, all_validation_samples, all_testing_samples, all_training_validation_samples


def reshape_data(original_data):
    """combine different samples for each features to one sample containing all features

    Args:
        original_data ([features * sample * sequence]): combination of samples for each feature

    Return:
        reshaped_data ([sample * sequence * features]): each sample contains myltiple features
    """
    result = []

    # [feature * sample] -> [sample * feature]
    samples = np.transpose(np.array(original_data, dtype=object), (1, 0))

    # [feature * sequence] -> [sequence * feature]

    for sample in samples:
        sample_data = []
        feature_num = len(sample)  # 6 features
        sequence_num = len(sample[0])  # number of steps in this sequence
        for i in range(sequence_num):
            sample_data.append([sample[j][i] for j in range(feature_num)])
        result.append(sample_data)
    return result


def dump_data(data, city, data_type):
    """save data as pickle file

    Args:
        data ([(feature)[(sample)[feature_id]]]): processed data
        city (str): city code for file naming
        data_type (str): data description for file naming
    """
    directory = './processed_data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = directory + "/{}_{}"

    pickle.dump(data, open(file_path.format(city, data_type), 'wb'))


def generate_poi_to_location(city_code, poi_mapping, input_data):
    # dataframe = pd.DataFrame(poi_mapping)
    # dataframe.to_csv(f"./raw_data/{city_code}_poi_mapping.csv", header="Id,VenueId")
    locations = []
    for venue_id in poi_mapping:
        record = input_data[input_data['VenueId'] == venue_id].iloc[0]
        longitude = record['Longitude']
        latitude = record['Latitude']
        locations.append([venue_id, longitude, latitude])
    df = pd.DataFrame(locations, columns=['POI_id', 'longitude', 'latitude'])
    df.to_csv(f"./raw_data/{city_code}_poi_mapping.csv")
    return df


# completely process data for one city

# def get_location_code_tile(latitude, longitude):
#     code_0 = latlon2quadkey(latitude, longitude, 12)
#     code_1 = latlon2quadkey(latitude, longitude, 13)
#     code_2 = latlon2quadkey(latitude, longitude, 14)
#     code_3 = latlon2quadkey(latitude, longitude, 15)
#     code_4 = latlon2quadkey(latitude, longitude, 16)
#     code_5 = latlon2quadkey(latitude, longitude, 17)
#     return [code_0, code_1, code_2, code_3, code_4, code_5]
#
#
# def get_location_code_h3(latitude, longitude):
#     code_0 = h3.latlng_to_cell(latitude, longitude, 5)
#     code_1 = h3.latlng_to_cell(latitude, longitude, 6)
#     code_2 = h3.latlng_to_cell(latitude, longitude, 7)
#     code_3 = h3.latlng_to_cell(latitude, longitude, 8)
#     code_4 = h3.latlng_to_cell(latitude, longitude, 9)
#     code_5 = h3.latlng_to_cell(latitude, longitude, 10)
#     return [code_0, code_1, code_2, code_3, code_4, code_5]
#
#
# def get_location_code_S2(latitude, longitude):
#     cell = S2.sphere.CellId().from_lat_lng(S2.sphere.LatLng.from_degrees(latitude, longitude))
#     code_0 = cell.parent(10).to_token()
#     code_1 = cell.parent(11).to_token()
#     code_2 = cell.parent(12).to_token()
#     code_3 = cell.parent(13).to_token()
#     code_4 = cell.parent(14).to_token()
#     code_5 = cell.parent(16).to_token()
#     return [code_0, code_1, code_2, code_3, code_4, code_5]

# pygeohash
# PHO    4(6)、5(60)、6(508) 、7(1075)、8(1367)     POI(1430)
# NYC    4(8)、5(67)、6(1042)、7(5310)、8(12927)    POI(15754)
# SIN    4(2)、5(24)、6(303) 、7(2615)、8(6273)      POI(8974)

# Bing Tile Map 编码
# PHO    12(22)、13(70)、14(197)、15(401)、16(654) 、17(894)     POI(1430)
# NYC    12(28)、13(86)、14(264)、15(765)、16(1843)、17(3484)    POI(15754)
# SIN    12(10)、13(24)、14(60) 、15(175)、16(512)、 17(1296)    POI(8974)

# PHO
#   tile 12(22)、13(70)、14(197)、15(401)、16(654)、17(894)    POI(1430)
#   h3 编码 5(7)、6(33)、7(155)、8(456)、9(792)、10(1104),11(1298),12(1382),13(1400),14、15(1405)    POI(1950)
#   S2 编码 9(7)、10(18)、11(58)、12(172)、13(391)、14(622),15(861),16(1085),17(1258),18(1347),19(1379)    POI(1950)
# NYC
#   h3 编码 5(8)、6(36)、7(183)、8(852)、9(2565)、10(5603)、11(10092)、12(13672)、13(15078)    POI(15754)
# SIN
#   h3 编码 5(5)、6(15)、7(60)、8(272)、9(1119)、10(2904)    POI(8974)
# def generate_area_dict(location_code_type, poi_sequences, poi_mapping, poi_location):
#     codes_0, codes_1, codes_2, codes_3, codes_4, codes_5 = [], [], [], [], [], []
#     dict_0, dict_1, dict_2, dict_3, dict_4, dict_5 = {}, {}, {}, {}, {}, {}
#     for sequence in poi_sequences:
#         sequence_0, sequence_1, sequence_2, sequence_3, sequence_4, sequence_5 = [], [], [], [], [], []
#         for seq in sequence:
#             seq_0, seq_1, seq_2, seq_3, seq_4, seq_5 = [], [], [], [], [], []
#             for poi in seq:
#                 poi_id = poi_mapping[poi]
#                 record = poi_location[poi_location['POI_id'] == poi_id].iloc[0]
#                 longitude = round(record['longitude'], 6)
#                 latitude = round(record['latitude'], 6)
#
#                 if location_code_type == "h3":
#                     code_0, code_1, code_2, code_3, code_4, code_5 = get_location_code_h3(latitude, longitude)
#                 elif location_code_type == "S2":
#                     code_0, code_1, code_2, code_3, code_4, code_5 = get_location_code_S2(latitude, longitude)
#                 elif location_code_type == "tile":
#                     code_0, code_1, code_2, code_3, code_4, code_5 = get_location_code_tile(latitude, longitude)
#                 else:
#                     raise Exception("Wrong location code type!")
#                 dict_0.setdefault(code_0, len(dict_0))
#                 dict_1.setdefault(code_1, len(dict_1))
#                 dict_2.setdefault(code_2, len(dict_2))
#                 dict_3.setdefault(code_3, len(dict_3))
#                 dict_4.setdefault(code_4, len(dict_4))
#                 dict_5.setdefault(code_5, len(dict_5))
#
#                 seq_0.append(dict_0[code_0])
#                 seq_1.append(dict_1[code_1])
#                 seq_2.append(dict_2[code_2])
#                 seq_3.append(dict_3[code_3])
#                 seq_4.append(dict_4[code_4])
#                 seq_5.append(dict_5[code_5])
#             sequence_0.append(seq_0)
#             sequence_1.append(seq_1)
#             sequence_2.append(seq_2)
#             sequence_3.append(seq_3)
#             sequence_4.append(seq_4)
#             sequence_5.append(seq_5)
#         codes_0.append(sequence_0)
#         codes_1.append(sequence_1)
#         codes_2.append(sequence_2)
#         codes_3.append(sequence_3)
#         codes_4.append(sequence_4)
#         codes_5.append(sequence_5)
#
#     codes_0 = np.array(codes_0)
#     codes_1 = np.array(codes_1)
#     codes_2 = np.array(codes_2)
#     codes_3 = np.array(codes_3)
#     codes_4 = np.array(codes_4)
#     codes_5 = np.array(codes_5)
#     area_dict = {"0": codes_0, "1": codes_1, "2": codes_2, "3": codes_3, "4": codes_4, "5": codes_5}
#
#     # 输出位置编码到 csv 中
#     # pd.DataFrame.from_dict(data=dict_0, orient='index').to_csv(f"./raw_data/PHO_S2_10.csv", header=False)
#     # pd.DataFrame.from_dict(data=dict_1, orient='index').to_csv(f"./raw_data/PHO_S2_11.csv", header=False)
#     # pd.DataFrame.from_dict(data=dict_2, orient='index').to_csv(f"./raw_data/PHO_S2_12.csv", header=False)
#     # pd.DataFrame.from_dict(data=dict_3, orient='index').to_csv(f"./raw_data/PHO_S2_13.csv", header=False)
#     # pd.DataFrame.from_dict(data=dict_4, orient='index').to_csv(f"./raw_data/PHO_S2_14.csv", header=False)
#     # pd.DataFrame.from_dict(data=dict_5, orient='index').to_csv(f"./raw_data/PHO_S2_16.csv", header=False)
#
#     return area_dict


def generate_h3_area_mapping():
    """
        Generate the index table of POI ID to area index
    """
    df = pd.read_csv(f"./raw_data/{city}_poi_mapping.csv")

    df["token_5"] = df.apply(lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], 5), axis=1)
    # Get the data in the h3_5 column and remove duplicates and sort
    token_5_values = df["token_5"].unique()
    token_5_values.sort()
    # Generate serial number
    index_5 = list(range(len(token_5_values)))
    index_5_dict = dict(zip(token_5_values, index_5))

    df['index_5'] = df['token_5'].map(index_5_dict)

    df["token_6"] = df.apply(lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], 6), axis=1)
    token_6_values = df["token_6"].unique()
    token_6_values.sort()
    index_6 = list(range(len(token_6_values)))
    index_6_dict = dict(zip(token_6_values, index_6))
    df['index_6'] = df['token_6'].map(index_6_dict)

    df["token_7"] = df.apply(lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], 7), axis=1)
    token_7_values = df["token_7"].unique()
    token_7_values.sort()
    index_7 = list(range(len(token_7_values)))
    index_7_dict = dict(zip(token_7_values, index_7))
    df['index_7'] = df['token_7'].map(index_7_dict)

    df["token_8"] = df.apply(lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], 8), axis=1)
    token_8_values = df["token_8"].unique()
    token_8_values.sort()
    index_8 = list(range(len(token_8_values)))
    index_8_dict = dict(zip(token_8_values, index_8))
    df['index_8'] = df['token_8'].map(index_8_dict)

    df["token_9"] = df.apply(lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], 9), axis=1)
    token_9_values = df["token_9"].unique()
    token_9_values.sort()
    index_9 = list(range(len(token_9_values)))
    index_9_dict = dict(zip(token_9_values, index_9))
    df['index_9'] = df['token_9'].map(index_9_dict)

    df["token_10"] = df.apply(lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], 10), axis=1)
    token_10_values = df["token_10"].unique()
    token_10_values.sort()
    index_10 = list(range(len(token_10_values)))
    index_10_dict = dict(zip(token_10_values, index_10))
    df['index_10'] = df['token_10'].map(index_10_dict)

    df.to_csv(f"./raw_data/{city}_h3_area_mapping.csv", index=False)
    print(f"Created {city}_h3_area_mapping.csv")


def get_area_index_h3(row):
    index_5 = row['index_5'].values[0]
    index_6 = row['index_6'].values[0]
    index_7 = row['index_7'].values[0]
    index_8 = row['index_8'].values[0]
    index_9 = row['index_9'].values[0]
    index_10 = row['index_10'].values[0]
    return [index_5, index_6, index_7, index_8, index_9, index_10]


def generate_area_dict_h3(location_code_type, poi_sequences, poi_mapping):
    codes_0, codes_1, codes_2, codes_3, codes_4, codes_5 = [], [], [], [], [], []

    area_mapping = pd.read_csv(f"./raw_data/{city}_{location_code_type}_area_mapping.csv")

    for sequence in poi_sequences:
        sequence_0, sequence_1, sequence_2, sequence_3, sequence_4, sequence_5 = [], [], [], [], [], []
        for seq in sequence:
            seq_0, seq_1, seq_2, seq_3, seq_4, seq_5 = [], [], [], [], [], []
            for poi in seq:
                poi_id = poi_mapping[poi]
                row = area_mapping.loc[area_mapping['POI_id'] == poi_id]
                if location_code_type == "h3":
                    index_5, index_6, index_7, index_8, index_9, index_10 = get_area_index_h3(row)
                # elif location_code_type == "S2":
                #     code_0, code_1, code_2, code_3, code_4, code_5 = get_location_code_S2(latitude, longitude)
                # elif location_code_type == "tile":
                #     code_0, code_1, code_2, code_3, code_4, code_5 = get_location_code_tile(latitude, longitude)
                else:
                    raise Exception("Wrong location code type!")

                seq_0.append(index_5)
                seq_1.append(index_6)
                seq_2.append(index_7)
                seq_3.append(index_8)
                seq_4.append(index_9)
                seq_5.append(index_10)
            sequence_0.append(seq_0)
            sequence_1.append(seq_1)
            sequence_2.append(seq_2)
            sequence_3.append(seq_3)
            sequence_4.append(seq_4)
            sequence_5.append(seq_5)
        codes_0.append(sequence_0)
        codes_1.append(sequence_1)
        codes_2.append(sequence_2)
        codes_3.append(sequence_3)
        codes_4.append(sequence_4)
        codes_5.append(sequence_5)

    codes_0 = np.array(codes_0)
    codes_1 = np.array(codes_1)
    codes_2 = np.array(codes_2)
    codes_3 = np.array(codes_3)
    codes_4 = np.array(codes_4)
    codes_5 = np.array(codes_5)
    area_dict = {"0": codes_0, "1": codes_1, "2": codes_2, "3": codes_3, "4": codes_4, "5": codes_5}

    return area_dict


def get_area_index_geohash(row):
    index_4 = row['index_4'].values[0]
    index_5 = row['index_5'].values[0]
    index_6 = row['index_6'].values[0]
    index_7 = row['index_7'].values[0]
    index_8 = row['index_8'].values[0]
    return [index_4, index_5, index_6, index_7, index_8]


def generate_area_dict_geohash(location_code_type, poi_sequences, poi_mapping):
    codes_0, codes_1, codes_2, codes_3, codes_4 = [], [], [], [], []

    area_mapping = pd.read_csv(f"./raw_data/{city}_{location_code_type}_area_mapping.csv")

    for sequence in poi_sequences:
        sequence_0, sequence_1, sequence_2, sequence_3, sequence_4 = [], [], [], [], []
        for seq in sequence:
            seq_0, seq_1, seq_2, seq_3, seq_4 = [], [], [], [], []
            for poi in seq:
                poi_id = poi_mapping[poi]
                row = area_mapping.loc[area_mapping['POI_id'] == poi_id]
                if location_code_type == "geohash":
                    index_4, index_5, index_6, index_7, index_8 = get_area_index_geohash(row)
                else:
                    raise Exception("Wrong location code type!")

                seq_0.append(index_4)
                seq_1.append(index_5)
                seq_2.append(index_6)
                seq_3.append(index_7)
                seq_4.append(index_8)
            sequence_0.append(seq_0)
            sequence_1.append(seq_1)
            sequence_2.append(seq_2)
            sequence_3.append(seq_3)
            sequence_4.append(seq_4)
        codes_0.append(sequence_0)
        codes_1.append(sequence_1)
        codes_2.append(sequence_2)
        codes_3.append(sequence_3)
        codes_4.append(sequence_4)

    codes_0 = np.array(codes_0)
    codes_1 = np.array(codes_1)
    codes_2 = np.array(codes_2)
    codes_3 = np.array(codes_3)
    codes_4 = np.array(codes_4)
    area_dict = {"0": codes_0, "1": codes_1, "2": codes_2, "3": codes_3, "4": codes_4}

    return area_dict


def generate_data(city):
    """
    Generate complete train and test data set for one city
        Save the result in pickle files
    Args:
        city (str): city to read data from and process
    """
    print(f"******Process data for {city}******")
    data = pd.read_csv(f"./raw_data/{city}_checkin_with_active_regionId.csv")

    visit_sequence_dict, total_sequences_meta = generate_sequence(data, min_seq_len, min_seq_num)
    valid_input_index = filter_long_short_term_sequences(total_sequences_meta, min_short_term_len, pre_seq_window,
                                                         min_long_term_count)
    # poi inputs
    poi_sequences, poi_mapping = generate_POI_sequences(data, visit_sequence_dict)
    # save
    with open(f'./processed_data/{city}_visit_sequence_dict.pickle', 'wb') as f:
        pickle.dump(visit_sequence_dict, f)
    with open(f'./processed_data/{city}_valid_input_index.pickle', 'wb') as f:
        pickle.dump(valid_input_index, f)
    np.save(f'./processed_data/{city}_poi_sequences.npy', poi_sequences)
    np.save(f'./processed_data/{city}_poi_mapping.npy', poi_mapping)

    # load
    with open(f'./processed_data/{city}_visit_sequence_dict.pickle', 'rb') as f:
        visit_sequence_dict = pickle.load(f)
    with open(f'./processed_data/{city}_valid_input_index.pickle', 'rb') as f:
        valid_input_index = pickle.load(f)
    poi_sequences = np.load(f'./processed_data/{city}_poi_sequences.npy', allow_pickle=True)
    poi_mapping = np.load(f'./processed_data/{city}_poi_mapping.npy', allow_pickle=True)

    # Export POI ID latitude and longitude
    # poi_location = generate_poi_to_location(city_code, poi_mapping, data)
    # print("poi mapping csv generated.")

    train_data, valid_data, test_data, train_valid_data, meta_data = [], [], [], [], {}

    poi_input_data = generate_input_samples(poi_sequences, valid_input_index)
    poi_train, poi_valid, poi_test, poi_train_valid = split_train_test(poi_input_data)
    train_data.append(poi_train)
    valid_data.append(poi_valid)
    test_data.append(poi_test)
    train_valid_data.append(poi_train_valid)
    print("poi sequence generated.")

    # cat inputs
    cat_sequences, cat_mapping = generate_category_sequences(data, visit_sequence_dict)
    cat_input_data = generate_input_samples(cat_sequences, valid_input_index)
    cat_train, cat_valid, cat_test, cat_train_valid = split_train_test(cat_input_data)
    train_data.append(cat_train)
    valid_data.append(cat_valid)
    test_data.append(cat_test)
    train_valid_data.append(cat_train_valid)
    print("category sequence generated.")

    # user inputs
    user_sequences, user_mapping = generate_user_sequences(data, visit_sequence_dict)
    user_input_data = generate_input_samples(user_sequences, valid_input_index)
    user_train, user_valid, user_test, user_train_valid = split_train_test(user_input_data)
    train_data.append(user_train)
    valid_data.append(user_valid)
    test_data.append(user_test)
    train_valid_data.append(user_train_valid)
    print("user sequence generated.")

    # hour inputs
    hour_sequences, hour_mapping = generate_hour_sequences(data, visit_sequence_dict)
    hour_input_data = generate_input_samples(hour_sequences, valid_input_index)
    hour_train, hour_valid, hour_test, hour_train_valid = split_train_test(hour_input_data)
    train_data.append(hour_train)
    valid_data.append(hour_valid)
    test_data.append(hour_test)
    train_valid_data.append(hour_train_valid)
    print("hour sequence generated.")

    # day inputs
    day_sequences, day_mapping = generate_day_sequences(data, visit_sequence_dict)
    day_input_data = generate_input_samples(day_sequences, valid_input_index)
    day_train, day_valid, day_test, day_train_valid = split_train_test(day_input_data)
    train_data.append(day_train)
    valid_data.append(day_valid)
    test_data.append(day_test)
    train_valid_data.append(day_train_valid)
    print("day sequence generated.")

    area_code_type = "geohash"  # geohash,h3,S2,tile
    area_dict = generate_area_dict_geohash(area_code_type, poi_sequences, poi_mapping)
    for key, area_data in area_dict.items():
        area_input_data = generate_input_samples(area_data, valid_input_index)
        area_train, area_valid, area_test, area_train_valid = split_train_test(area_input_data)
        train_data.append(area_train)
        valid_data.append(area_valid)
        test_data.append(area_test)
        train_valid_data.append(area_train_valid)
    print(f"all {area_code_type} area sequence generated.")

    # reshape data: [features * sample * sequence] -> [sample * sequence * features]
    train_data = reshape_data(train_data)
    valid_data = reshape_data(valid_data)
    test_data = reshape_data(test_data)
    train_valid_data = reshape_data(train_valid_data)

    # meta data
    meta_data["POI"] = poi_mapping
    meta_data["cat"] = cat_mapping
    meta_data["user"] = user_mapping
    meta_data["hour"] = hour_mapping
    meta_data["day"] = day_mapping

    # output data
    dump_data(train_data, city, "train")
    dump_data(valid_data, city, "valid")
    dump_data(test_data, city, "test")
    dump_data(train_valid_data, city, "train_valid")
    dump_data(meta_data, city, "meta")


if __name__ == '__main__':
    # city_list = ['PHO', 'NYC', 'SIN']
    city_list = ['PHO']
    # generate_h3_area_mapping()

    for city in city_list:
        generate_data(city)
