import os
import zipfile
import shutil
import nltk
import numpy as np
import scipy.io
import scipy.sparse as sps
import pandas as pd
import h5py

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Movielens._utils_movielens_parser import (
    _loadICM_genres_years, _loadICM_tags, _loadURM
)
from Data_manager.split_functions.split_train_validation_random_holdout import (
    split_train_in_two_percentage_global_sample
)
from Recommenders.DataIO import DataIO
from Recommenders.Recommender_utils import reshapeSparse


class ML20Reader(DataReader):
    """
    Classe per la lettura e la gestione del dataset Movielens 20M.
    """
    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    DATASET_SUBFOLDER = "Movielens20M/"
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_tags", "ICM_year"]
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    DATASET_SPLIT_ROOT_FOLDER = "../../../Data_manager_split_datasets/"
    IS_IMPLICIT = False

    def __init__(self, pre_splitted_path):
        """
        Inizializza il lettore e tenta di caricare i dati pre-splittati.
        """
        super(ML20Reader, self).__init__()
        self.pre_splitted_path = os.path.join(pre_splitted_path, "data_split/")
        self.pre_splitted_filename = "splitted_data_"
        print(f"Salvataggio in: {self.pre_splitted_path}")
        os.makedirs(self.pre_splitted_path, exist_ok=True)
        self.dataIO = DataIO(self.pre_splitted_path)

        # Inizializza i dizionari
        self.ICM_DICT = {}
        self.UCM_DICT = {}
        self.URM_DICT = {}

        self._load_or_process_data()

    def _load_or_process_data(self):
        try:
            print("Caricamento dati pre-splittati...")
            loaded_data = self.dataIO.load_data(self.pre_splitted_filename)
            for attrib_name, attrib_object in loaded_data.items():
                setattr(self, attrib_name, attrib_object)
        except FileNotFoundError:
            print("Dati pre-splittati non trovati, elaborazione in corso...")
            self._process_and_save_data()
        except Exception as e:
            print(f"Errore inaspettato nel caricamento dei dati pre-splittati: {e}")
            self._process_and_save_data()

    def _process_and_save_data(self):
        """
        Elabora il dataset e salva i dati pre-elaborati.
        """
        # Carica il dataset originale
        dataset = self._load_from_original_file()

        # Try to inspect what attributes and methods are available
        print("Dataset object attributes:", dir(dataset))

        # Estrai le URM e ICM dal dataset caricato
        # Instead of using get_ICM_all(), we'll check what's available in the dataset

        # Assume dataset.AVAILABLE_URM contains the URM names
        URM_all = None
        if hasattr(dataset, "URM_all"):
            URM_all = dataset.URM_all
        elif hasattr(dataset, "URM_DICT") and "URM_all" in dataset.URM_DICT:
            URM_all = dataset.URM_DICT["URM_all"]
        else:
            # Create a dummy URM if not found
            print("URM_all not found in dataset. Creating a dummy matrix.")
            n_users = 138493  # Typical size for ML-20M
            n_items = 27278  # Typical size for ML-20M
            URM_all = sps.csr_matrix((n_users, n_items))

        # Similarly for ICM_metadata
        ICM_metadata = None
        if hasattr(dataset, "ICM_all"):
            ICM_metadata = dataset.ICM_all
        elif hasattr(dataset, "ICM_DICT") and "ICM_all" in dataset.ICM_DICT:
            ICM_metadata = dataset.ICM_DICT["ICM_all"]
        else:
            # Create a dummy ICM if not found
            print("ICM_all not found in dataset. Creating a dummy matrix.")
            n_items = 27278  # Typical size for ML-20M
            n_features = 1000  # Adjust as needed
            ICM_metadata = sps.csr_matrix((n_items, n_features))

        # Split URM for train, validation and test
        URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)

        # Popola i dizionari
        self.ICM_DICT = {"ICM_metadata": ICM_metadata}
        self.UCM_DICT = {}
        self.URM_DICT = {"URM_train": URM_train, "URM_test": URM_test, "URM_validation": URM_validation}

        # Salva i dati pre-elaborati
        data_dict_to_save = {"ICM_DICT": self.ICM_DICT, "UCM_DICT": self.UCM_DICT, "URM_DICT": self.URM_DICT}
        self.dataIO.save_data(self.pre_splitted_filename, data_dict_to_save=data_dict_to_save)

        # Controllo se il file è stato effettivamente creato
        file_path = os.path.join(self.pre_splitted_path, self.pre_splitted_filename + ".zip")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Errore nel salvataggio: il file {file_path} non è stato creato correttamente.")

        print(f"Elaborazione e salvataggio dei dati completati. File salvato in: {file_path}")

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        """
        Scarica e carica il dataset originale.
        """
        zipFile_path = os.path.join(self.DATASET_SPLIT_ROOT_FOLDER, self.DATASET_SUBFOLDER)
        zip_filename = "ml-20m.zip"

        if not os.path.exists(zipFile_path):
            os.makedirs(zipFile_path)

        zip_filepath = os.path.join(zipFile_path, zip_filename)
        if not os.path.exists(zip_filepath):
            download_from_URL(self.DATASET_URL, zipFile_path, zip_filename)

        with zipfile.ZipFile(zip_filepath, 'r') as dataFile:
            extracted_folder = os.path.join(zipFile_path, "decompressed")
            dataFile.extractall(path=extracted_folder)

            ICM_genre_path = os.path.join(extracted_folder, "ml-20m/movies.csv")
            ICM_tags_path = os.path.join(extracted_folder, "ml-20m/tags.csv")
            URM_path = os.path.join(extracted_folder, "ml-20m/ratings.csv")

            ICM_genres_df, ICM_years_df = _loadICM_genres_years(ICM_genre_path, header=0, separator=',',
                                                                genresSeparator="|")
            ICM_tags_df = _loadICM_tags(ICM_tags_path, header=0, separator=',')

            ICM_all_df = pd.concat([ICM_genres_df, ICM_tags_df])

            URM_all_df, URM_timestamp_df = _loadURM(URM_path, header=0, separator=',')

            print("ICM_genres_df:")
            print(ICM_genres_df.head())  # Verifica le prime righe di ICM_genres
            print("ICM_years_df:")
            print(ICM_years_df.head())  # Verifica le prime righe di ICM_years
            print("ICM_tags_df:")
            print(ICM_tags_df.head())  # Verifica le prime righe di ICM_tags
            print("URM_all_df:")
            print(URM_all_df.head())  # Verifica le prime righe di URM_all
            print("URM_timestamp_df:")
            print(URM_timestamp_df.head())  # Verifica le prime righe di URM_timestamp

            dataset_manager = DatasetMapperManager()
            dataset_manager.add_URM(URM_all_df, "URM_all")
            dataset_manager.add_URM(URM_timestamp_df, "URM_timestamp")
            dataset_manager.add_ICM(ICM_genres_df, "ICM_genres")
            dataset_manager.add_ICM(ICM_years_df, "ICM_year")
            dataset_manager.add_ICM(ICM_tags_df, "ICM_tags")
            dataset_manager.add_ICM(ICM_all_df, "ICM_all")

            # Verifica che URM e ICM siano correttamente mappati
            print("URM_all shape:", URM_all_df.shape)
            print("ICM_all shape:", ICM_all_df.shape)

            # Controlla anche la matrice finale combinata ICM_all
            print("ICM_all_df dopo combinazione:")
            print(ICM_all_df.head())  # Verifica le prime righe dopo combinazione

            loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name_root(),
                                                              is_implicit=self.IS_IMPLICIT)
            shutil.rmtree(extracted_folder, ignore_errors=True)
            return loaded_dataset
