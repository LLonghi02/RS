#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/01/18

@author: Maurizio Ferrari Dacrema
"""


import ast, gzip, os
import pandas as pd
import numpy as np
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, remove_Dataframe_duplicates

def parse_txt(filename):
    f = gzip.open(filename, 'r', encoding='ISO-8859-1') if filename.split('.')[-1] == 'gz' else open(filename, 'r', encoding='ISO-8859-1')

    full_entry = {}
    last_key = None
    last_value = None

    expected_key = ["product/productId",
                 "review/userId",
                 "review/profileName",
                 "review/helpfulness",
                 "review/score",
                 "review/time",
                 "review/summary",
                 "review/text",
                ]
    expected_index = 0

    for l in f:
        l = l.strip()
        splitted = l.split(':')

        if len(splitted) >= 2 and expected_index < len(expected_key) and splitted[0] == expected_key[expected_index]:
            # If the line has the ":", there are still keys I am expecting and the string begins with one, proceed reading
            if last_key is not None:
                full_entry[last_key] = last_value
            last_key = expected_key[expected_index]
            last_value = l[len(last_key)+2:]
            expected_index += 1

        elif expected_index == len(expected_key) and l == "":
            # The full entry is complete at the first newline once all the keys have been found
            full_entry[last_key] = last_value
            yield full_entry

            # Reset for new entry
            full_entry = {}
            last_key = None
            last_value = None
            expected_index = 0
        else:
            #Incorrect line, possibly a "\n" in the profileName field
            print('Erroneous newline detected, merging string "{}" with previous key: "{}"'.format(l, last_key))
            last_value += " " + l

def parse_json(file_path):
    g = open(file_path, 'r')

    for l in g:
        try:
            yield ast.literal_eval(l)
        except Exception as exception:
            print("Exception: {}. Skipping".format(str(exception)))



class _AmazonReviewDataReader(DataReader):
    """
    This Class refers to the Amazon Product Data dataset collection
    https://jmcauley.ucsd.edu/data/amazon/

    This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014.

    This dataset includes:
        reviews (ratings, text, helpfulness votes),
        product metadata (descriptions, category information, price, brand, and image features),
        links (also viewed/also bought graphs).

    """

    DATASET_SUBFOLDER = "AmazonReviewData/"

    IS_IMPLICIT = False


    def _get_ICM_metadata_path(self, data_folder, compressed_file_name, decompressed_file_name, file_url):
        """
        Metadata files are .csv
        :param data_folder:
        :param file_name:
        :param file_url:
        :return:
        """


        try:

            open(data_folder + decompressed_file_name, "r")

        except FileNotFoundError:

            self._print("Decompressing metadata file...")

            try:

                decompressed_file = open(data_folder + decompressed_file_name, "wb")

                compressed_file = gzip.open(data_folder + compressed_file_name, "rb")
                decompressed_file.write(compressed_file.read())

                compressed_file.close()
                decompressed_file.close()

            except (FileNotFoundError, Exception):

                self._print("Unable to find or decompress compressed file. Downloading...")

                download_from_URL(file_url, data_folder, compressed_file_name)

                decompressed_file = open(data_folder + decompressed_file_name, "wb")

                compressed_file = gzip.open(data_folder + compressed_file_name, "rb")
                decompressed_file.write(compressed_file.read())

                compressed_file.close()
                decompressed_file.close()


        return data_folder + decompressed_file_name






    def _get_URM_review_path(self, data_folder, file_name, file_url):
        """
        Metadata files are .csv
        :param data_folder:
        :param file_name:
        :param file_url:
        :return:
        """


        try:

            open(data_folder + file_name, "r")

        except FileNotFoundError:

            self._print("Unable to find or open review file. Downloading...")
            download_from_URL(file_url, data_folder, file_name)


        return data_folder + file_name



    def _load_from_original_file_all_amazon_datasets(self, URM_path, metadata_path = None, reviews_path = None, txt_format : bool = False):
        # Load data from original
        dataset_manager = DatasetMapperManager()

        if txt_format:
            self._print("Loading Interactions")
            parsed_file = URM_path.replace(".txt", ".json")

            if os.path.exists(parsed_file):
                reviews_df = pd.read_json(parsed_file)

            else:
                reviews_df = []
                for entry in parse_txt(URM_path):
                    reviews_df.append(entry)

                reviews_df = pd.DataFrame(reviews_df)
                reviews_df.to_json(parsed_file)

            URM_all_dataframe = reviews_df.loc[:,['review/userId', 'product/productId', 'review/score', 'review/time']]
            URM_all_dataframe.columns = ["UserID", "ItemID", "Interaction", "Timestamp"]
            URM_all_dataframe.loc[:, ['UserID', 'ItemID']] = URM_all_dataframe.loc[:, ['UserID', 'ItemID']].astype(str)

            # Remove duplicates selecting the most recent interaction
            URM_all_dataframe["Timestamp"] = URM_all_dataframe["Timestamp"].astype(int)
            URM_all_dataframe = remove_Dataframe_duplicates(URM_all_dataframe,
                                                            unique_values_in_columns=['UserID', 'ItemID'],
                                                            keep_highest_value_in_col="Timestamp")

        else:
            self._print("Loading Interactions")
            URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep=",", header=None, dtype={0:str, 1:str, 2:float, 3:int})
            URM_all_dataframe.columns = ["UserID", "ItemID", "Interaction", "Timestamp"]

        URM_timestamp_dataframe = URM_all_dataframe.copy().drop(columns=["Interaction"])
        URM_all_dataframe = URM_all_dataframe.drop(columns=["Timestamp"])
        URM_timestamp_dataframe.columns = ["UserID", "ItemID", "Data"]
        URM_all_dataframe.columns = ["UserID", "ItemID", "Data"]

        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_timestamp_dataframe, "URM_timestamp")

        if metadata_path is not None and not txt_format:
            self._print("Loading Item Features Metadata")
            ICM_metadata_dataframe = self._loadMetadata(metadata_path)
            dataset_manager.add_ICM(ICM_metadata_dataframe, "ICM_metadata")

        if reviews_path is not None and not txt_format:
            self._print("Loading Item Features Reviews")
            ICM_reviews_dataframe = self._loadReviews(reviews_path)
            dataset_manager.add_ICM(ICM_reviews_dataframe, "ICM_reviews")


        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)


        # Clean temp files
        self._print("Cleaning Temporary Files")

        if metadata_path is not None:
            os.remove(metadata_path)

        if reviews_path is not None:
            os.remove(reviews_path)

        self._print("Loading Complete")

        return loaded_dataset







    def _loadMetadata(self, file_path):

        from Data_manager.TagPreprocessing import tagFilterAndStemming
        import itertools


        parser_metadata = parse_json(file_path)


        def _get_tag_list(new_metadata):

            # Categories are a list of lists. Unclear whether only the first elements contains data or not
            item_categories = new_metadata["categories"]
            if not np.all(pd.isna(item_categories)):
                item_categories = list(itertools.chain.from_iterable(item_categories))
            else:
                item_categories = []


            token_list = list([new_metadata['title'] if "title" in new_metadata else [],
                              new_metadata['brand'] if "brand" in new_metadata else [],
                              *item_categories,
                              new_metadata['description'] if "description" in new_metadata else []])

            # Sometimes brand or other fields are not present (nan) or empty ('')
            token_list = [token for token in token_list if not pd.isna(token) and len(token)>0]
            token_list = ' '.join(token_list)

            # Remove non alphabetical character and split on spaces
            token_list = tagFilterAndStemming(token_list)

            # Remove duplicates
            token_list = list(set(token_list))

            return token_list

        tag_lists = []
        item_id_list = []

        for index, new_metadata in enumerate(parser_metadata):

            if (index % 10000 == 0 and index > 0):
                self._print("Processed {}".format(index))

            # Put the feature I want in a list, which will be considered as a "TagList"
            this_tag_list = _get_tag_list(new_metadata)
            item_id_list.append(new_metadata["asin"])
            tag_lists.append(this_tag_list)

        ICM_metadata_dataframe = pd.DataFrame({"ItemID": item_id_list,
                                               "FeatureID_list": tag_lists
                                               })

        ICM_metadata_dataframe = ICM_metadata_dataframe.explode("FeatureID_list")
        ICM_metadata_dataframe.rename(columns={"FeatureID_list": "FeatureID"}, inplace=True)
        ICM_metadata_dataframe["Data"] = 1


        return ICM_metadata_dataframe



    def _loadReviews(self, file_path):

        from Data_manager.TagPreprocessing import tagFilterAndStemming, tagFilter


        parser_reviews = parse_json(file_path)

        item_id_list = []
        tag_lists = []

        for index, new_review in enumerate(parser_reviews):

            if (index % 10000 == 0 and index > 0):
                self._print("Processed {}".format(index))

            # Put the feature I want in a list, which will be considered as a "TagList"
            reviewText = new_review["reviewText"]
            reviewSummary = new_review["summary"]

            this_tag_list = ' '.join([reviewText, reviewSummary])

            item_id_list.extend([new_review["asin"]]*len(this_tag_list))
            tag_lists.extend(this_tag_list)

        ICM_reviews_dataframe = pd.DataFrame({"ItemID": item_id_list,
                                              "FeatureID": tag_lists,
                                              "Data": [1]*len(item_id_list),
                                              })

        return ICM_reviews_dataframe
