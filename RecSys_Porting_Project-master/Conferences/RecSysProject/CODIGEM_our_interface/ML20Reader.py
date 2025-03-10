#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""
from Conferences.RecSysProject.CODIGEM_github.data_processing import DataLoader

from Recommenders.DataIO import DataIO
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import os

from Data_manager.Movielens.Movielens20MReader import Movielens20MReader as Movielens20MReader_DataManager

#TODO integra il main qui
class ML20Reader(object):

    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(ML20Reader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"
        path = os.path.join(os.path.dirname(__file__), '..','..','..', 'Data_manager', 'Movielens', 'ml-20m',
                                    'ratings.csv')

        pro_dir = os.path.join(path, 'pro_sg')

        if os.path.exists(pro_dir):
            print("Data Already Preprocessed")
            loader = DataLoader(path)
            n_items = loader.load_n_items()
            train_data = loader.load_data('train')
            vad_data_tr, vad_data_te = loader.load_data('validation')
            test_data_tr, test_data_te = loader.load_data('test')

        else:
            print("Data Not Preprocessed")
            print("Preprocessing Data")
            current_directory = os.path.dirname(__file__)
            data_processing_path = os.path.join(current_directory, '..', '..', '..', 'Conferences', 'RecSysProject',
                                                'CODIGEM_github', 'data_processing.py')
            os.system(f'python {data_processing_path}')
            loader = DataLoader(path)
            n_items = loader.load_n_items()
            train_data = loader.load_data('train')
            vad_data_tr, vad_data_te = loader.load_data('validation')
            test_data_tr, test_data_te = loader.load_data('test')




