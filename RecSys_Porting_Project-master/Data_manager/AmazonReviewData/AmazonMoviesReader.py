#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27/02/24

@author: Andrea Pisani
"""



from Data_manager.AmazonReviewData._AmazonReviewDataReader import _AmazonReviewDataReader



class AmazonMoviesReader(_AmazonReviewDataReader):
    """
    From SNAP
    https://snap.stanford.edu/data/web-Movies.html

    This dataset consists of movie reviews from amazon. The data span a period of more than 10 years,
    including all ~8 million reviews up to October 2012. Reviews include product and user information,
    ratings, and a plaintext review. We also have reviews from all other Amazon categories.

    Citation
    J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through
    online reviews. WWW, 2013.

    """


    DATASET_URL_RATING = "https://snap.stanford.edu/data/movies.txt.gz"
    DATASET_URL_METADATA = "https://snap.stanford.edu/data/movies.txt.gz"
    DATASET_URL_REVIEWS = "https://snap.stanford.edu/data/movies.txt.gz"

    DATASET_SUBFOLDER = "AmazonReviewData/AmazonMovies/"
    AVAILABLE_ICM = []


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        dataset_split_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        metadata_path = self._get_ICM_metadata_path(data_folder = dataset_split_folder,
                                                    compressed_file_name = "movies.txt.gz",
                                                    decompressed_file_name = "movies.txt",
                                                    file_url = self.DATASET_URL_METADATA)

        # reviews_path = self._get_ICM_metadata_path(data_folder=dataset_split_folder,
        #                                            compressed_file_name="movies.txt.gz",
        #                                            decompressed_file_name="movies.txt",
        #                                            file_url=self.DATASET_URL_REVIEWS)


        #URM_path = self._get_URM_review_path(data_folder = dataset_split_folder,
        #                                     file_name = "movies.txt.gz",
        #                                     file_url = self.DATASET_URL_RATING)


        loaded_dataset = self._load_from_original_file_all_amazon_datasets(metadata_path, txt_format = True)
                                                                           #metadata_path = metadata_path,
                                                                           #reviews_path = reviews_path,
                                                                           # txt = True)

        return loaded_dataset

