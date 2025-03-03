#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Data_manager.AmazonReviewData._AmazonReviewDataReader import _AmazonReviewDataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

import scipy.io
import scipy.sparse as sps
import h5py, os
import numpy as np

from Recommenders.DataIO import DataIO
from Recommenders.Recommender_utils import reshapeSparse

class AEReader(_AmazonReviewDataReader):

    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    DATASET_URL_RATING = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv"
    DATASET_URL_METADATA = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz"

    DATASET_SUBFOLDER = "AmazonReviewData/AmazonElectronics/"
    AVAILABLE_ICM = ["ICM_metadata"]

    def __init__(self, pre_splitted_path):

        super(AEReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        #original_data_path = os.path.join("C:", "Users", "laral", "Desktop", "RS", "RecSys_Porting_Project-master",
                                          #"Data_manager", "AmazonReviewData", "AmazonElectronicsReader.py")

        # original_data_path = os.path.join("/Users", "michelebalena", "Documents", "GitHub", "RS",
        #                                   "RecSys_Porting_Project-master", "Data_manager",
        #                                   "AmazonReviewData", "AmazonElectronicsReader.py")
        # print("Percorso del dataset:", original_data_path)

        import os

        original_data_path = os.path.join("/Users", "michelebalena", "Documents", "GitHub", "RS",
                                          "RecSys_Porting_Project-master", "Data_manager",
                                          "AmazonReviewData", "AmazonElectronicsReader.py")

        if not os.path.exists(original_data_path):
            print(f"Errore: Il file {original_data_path} non esiste.")
            return


        # Se la cartella di dati pre-split non esiste, la crea
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)
        print(dataIO)

        try:
            print("AmazonElectronicsReader: Tentativo di caricare i dati pre-splittati")
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:
            print("AmazonElectronicsReader: Dati pre-splittati non trovati, costruendo nuovi dati")
        except (OSError, IOError) as e:
            print(f"Errore nell'accesso ai file pre-splittati: {e}")
        except Exception as e:
            print(f"Errore imprevisto nel caricamento dei dati: {e}")

            print("AmazonElectronicsReader: Caricamento del URM")

            metadata_path = self._get_ICM_metadata_path(data_folder=original_data_path,
                                                       compressed_file_name="meta_Electronics.json.gz",
                                                       decompressed_file_name="meta_Electronics.json",
                                                       file_url=self.DATASET_URL_METADATA)

            URM_path = self._get_URM_review_path(data_folder=original_data_path,
                                                 file_name="ratings_Electronics.csv",
                                                 file_url=self.DATASET_URL_RATING)

            loaded_dataset = self._load_from_original_file_all_amazon_datasets(URM_path,
                                                                               metadata_path=metadata_path,
                                                                               reviews_path=None)

            URM_train_builder = loaded_dataset['URM_train']
            URM_test_builder = loaded_dataset['URM_test']

            # Caricamento della matrice ICM per le caratteristiche degli articoli
            ICM_metadata = scipy.io.loadmat(metadata_path)['X']
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

            print("AmazonElectronicsReader: Caricamento completato")

    def _load_from_original_file(self):
        # Metodo per il caricamento dei dati originali (già implementato in _AmazonReviewDataReader)
        pass


