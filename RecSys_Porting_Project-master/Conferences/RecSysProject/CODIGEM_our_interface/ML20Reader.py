#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.DataIO import DataIO
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import os

from Data_manager.Movielens.Movielens20MReader import Movielens20MReader as Movielens20MReader_DataManager


class ML20MReader(object):

    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(ML20MReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:

            print("ML20MReader: Attempting to load pre-splitted data (Reader)")

            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("ML20MReader: Pre-splitted data not found, building new one")

            print("ML20MReader: loading URM")

            # TODO Replace this with the publicly available dataset you need
            #  The DataManagers are in the Data_Manager folder, if the dataset is already there use that data reader
            data_reader = Movielens20MReader_DataManager()
            dataset = data_reader.load_data()

            URM_all = dataset.get_URM_all()

            # TODO Apply data preprocessing if required (for example binarizing the data, removing users ...)
            # Selezione dei punteggi: mantenere solo valutazioni >= 3 e convertire il resto in 1
            URM_all.data = (URM_all.data >= 3).astype(int)

            # Filtraggio di utenti e oggetti poco attivi
            from scipy.sparse import csr_matrix
            import numpy as np

            def filter_users_and_items(URM, min_interactions=10):
                user_interactions = np.array((URM > 0).sum(axis=1)).flatten()
                item_interactions = np.array((URM > 0).sum(axis=0)).flatten()

                users_to_keep = np.where(user_interactions >= min_interactions)[0]
                items_to_keep = np.where(item_interactions >= min_interactions)[0]

                URM = URM[users_to_keep, :]
                URM = URM[:, items_to_keep]

                return URM

            URM_all = filter_users_and_items(URM_all, min_interactions=10)
            URM_all.eliminate_zeros()

            # TODO select the data splitting that you need, almost certainly there already is a function that does the splitting
            # Split the data in train, validation and test
            URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.8)


            # TODO get the sparse matrices in the correct dictionary with the correct name
            # TODO ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required
            self.ICM_DICT = {}
            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }


            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("ML20MReader: loading complete")



