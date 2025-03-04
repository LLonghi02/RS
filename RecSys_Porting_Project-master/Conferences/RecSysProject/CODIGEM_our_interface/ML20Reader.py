#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Data_manager.AmazonReviewData._AmazonReviewDataReader import _AmazonReviewDataReader
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Movielens._utils_movielens_parser import _loadICM_genres_years, _loadICM_tags, _loadURM
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

import zipfile
import pandas as pd
import shutil

import scipy.io
import scipy.sparse as sps
import h5py, os
import numpy as np

from Recommenders.DataIO import DataIO
from Recommenders.Recommender_utils import reshapeSparse

class ML20MReader(DataReader):

    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    DATASET_SUBFOLDER = "Movielens20M/"
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_tags", "ICM_year"]
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]

    IS_IMPLICIT = False

    def __init__(self, pre_splitted_path):

        super(ML20MReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        original_data_path = os.path.join(os.path.dirname(__file__), '..', "..",
                                          "RecSys_Porting_Project-master/Data_manager/Movielens")

        if not os.path.exists(original_data_path):
            print(f"Errore: Il file {original_data_path} non esiste.")
            return


        # Se la cartella di dati pre-split non esiste, la crea
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)
        loaded_data = dataIO.load_data(pre_splitted_filename)
        print("Dati caricati:", loaded_data)  # Controlla cosa viene caricato

        try:
            print("ML20Reader: Tentativo di caricare i dati pre-splittati")
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:
            print("ML20Reader: Dati pre-splittati non trovati, costruendo nuovi dati")
        except (OSError, IOError) as e:
            print(f"Errore nell'accesso ai file pre-splittati: {e}")
        except Exception as e:
            print(f"Errore imprevisto nel caricamento dei dati: {e}")

            print("ML20Reader: Caricamento del URM")

            URM_train_builder = loaded_data['URM_train']
            URM_test_builder = loaded_data['URM_test']

            # Caricamento della matrice ICM per le caratteristiche degli articoli
            ICM_metadata = scipy.io.loadmat()['X']
            ICM_metadata = sps.csr_matrix(ICM_metadata)

            # Matrimonio booleano per la matrice ICM
            ICM_metadata_bool = ICM_metadata.copy()
            ICM_metadata_bool.data = np.ones_like(ICM_metadata_bool.data)

            # Adattamento delle dimensioni delle matrici
            n_rows = max(URM_test_builder.shape[0], URM_train_builder.shape[0])
            n_cols = max(URM_test_builder.shape[1], URM_train_builder.shape[1], ICM_metadata.shape[0])

            newShape = (n_rows, n_cols)

            URM_test = reshapeSparse(URM_test_builder, newShape)
            URM_train = reshapeSparse(URM_train_builder, newShape)

            # Divisione dei dati di addestramento in train e validazione
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train.copy(), train_percentage=0.8)

            # Aggiunta delle matrici al dizionario
            self.ICM_DICT = {
                "ICM_metadata": ICM_metadata,
            }

            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }

            # Salvataggio dei dati pre-elaborati
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("ML20Reader: Caricamento completato")

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original

        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-20m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ml-20m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-20m.zip")

        ICM_genre_path = dataFile.extract("ml-20m/movies.csv", path=zipFile_path + "decompressed/")
        ICM_tags_path = dataFile.extract("ml-20m/tags.csv", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-20m/ratings.csv", path=zipFile_path + "decompressed/")

        self._print("Loading Item Features Genres")
        ICM_genres_dataframe, ICM_years_dataframe = _loadICM_genres_years(ICM_genre_path, header=0, separator=',',
                                                                          genresSeparator="|")

        self._print("Loading Item Features Tags")
        ICM_tags_dataframe = _loadICM_tags(ICM_tags_path, header=0, separator=',')

        ICM_all_dataframe = pd.concat([ICM_genres_dataframe, ICM_tags_dataframe])

        self._print("Loading Interactions")
        URM_all_dataframe, URM_timestamp_dataframe = _loadURM(URM_path, header=0, separator=',')

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_timestamp_dataframe, "URM_timestamp")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_genres")
        dataset_manager.add_ICM(ICM_years_dataframe, "ICM_year")
        dataset_manager.add_ICM(ICM_tags_dataframe, "ICM_tags")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("saving URM and ICM")
        print ("ho fatto il reading")
        return loaded_dataset


