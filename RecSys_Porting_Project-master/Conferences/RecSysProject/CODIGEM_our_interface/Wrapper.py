import numpy as np
import tensorflow as tf
import scipy.sparse as sps
import logging
import os

from Conferences.RecSysProject.CODIGEM_github.ddgm_model_rs import DDGM
from Conferences.RecSysProject.CODIGEM_github.main import n_items, train_data
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.DataIO import DataIO

import torch.nn as nn
import pandas as pd
import Conferences.RecSysProject.CODIGEM_github.ddgm_model_rs as dg


#NOTA è univoco per tutti i dataset
class CODIGEM_Wrapper(Incremental_Training_Early_Stopping, BaseTempFolder):
    """
    Wrapper per il modello di raccomandazione ML20.
    Implementa un sistema basato su deep learning utilizzando TensorFlow e il modello DDGM.
    """

    RECOMMENDER_NAME = "CODIGEM_Wrapper"

    def __init__(self, URM_train):
        """
        Inizializza il wrapper con la matrice di interazioni utente-elemento (URM).
        """
        super(CODIGEM_Wrapper, self).__init__(URM_train)
        self.URM_train = sps.csr_matrix(URM_train)  # Convertiamo direttamente in CSR una sola volta
        self._item_indices = np.arange(self.n_items, dtype=np.int32)
        self.sess = None  # Inizializzazione della sessione TensorFlow

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        Calcola il punteggio degli item per ciascun utente dato.
        """
        item_scores = -np.ones((len(user_id_array), self.n_items)) * np.inf
        item_indices = items_to_compute if items_to_compute is not None else self._item_indices

        for user_index, user_id in enumerate(user_id_array):
            item_score_user = self.model.predict(user_id, item_indices)
            item_scores[user_index, item_indices] = item_score_user.ravel()

        return item_scores

    def _init_model(self):
        """
        Inizializza il modello DDGM e crea una nuova sessione TensorFlow.
        """
        if self.sess:
            self.sess.close()

        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.Session()

#TODO da cambiare va messo quello che fa in C:\Users\laral\Desktop\RS\RecSys_Porting_Project-master\Conferences\RecSysProject\CODIGEM_github\ddgm_model_rs.py

        # Parameters related to the model

        num = train_data.shape[0]  # number of rows in the dataframe
        D = n_items  # input dimension
        M = 200  # the number of neurons in scale (s) and translation (t) nets
        T = 3  # hyperparater to tune
        beta = 0.0001  # hyperparater to tune #Beta = 0.0001 is best so far
        # Initializing the model

        p_dnns = nn.ModuleList([nn.Sequential(nn.Linear(D, M), nn.PReLU(),
                                              nn.Linear(M, M), nn.PReLU(),
                                              nn.Linear(M, M), nn.PReLU(),
                                              nn.Linear(M, M), nn.PReLU(),
                                              nn.Linear(M, M), nn.PReLU(),
                                              nn.Linear(M, 2 * D)) for _ in range(T - 1)])

        decoder_net = nn.Sequential(nn.Linear(D, M), nn.PReLU(),
                                    nn.Linear(M, M), nn.PReLU(),
                                    nn.Linear(M, M), nn.PReLU(),
                                    nn.Linear(M, M), nn.PReLU(),
                                    nn.Linear(M, M), nn.PReLU(),
                                    nn.Linear(M, D), nn.Tanh())

        self.model = dg.DDGM(p_dnns, decoder_net, beta=beta, T=T, D=D)


#TODO vanno messi i valori del report li trovi in CODIGEM_github/PAPER
    def fit(self,
            epochs=100,
            num_factors=64,
            batch_size=200,
            learning_rate=0.001,

            # Parametri standard
            temp_file_folder=None,
            **earlystopping_kwargs
            ):

        """
        Addestra il modello utilizzando early stopping.
        """
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        # Inizializzazione parametri
        self.learning_rate = learning_rate
        self.num_factors = num_factors
        self.batch_size = batch_size

        self._init_model()
        self._update_best_model()
        self._train_with_early_stopping(epochs, algorithm_name=self.RECOMMENDER_NAME, **earlystopping_kwargs)
        self.load_model(self.temp_file_folder, file_name="_best_model")
        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

    def _run_epoch(self, currentEpoch):
        """
        Esegue un'epoca di training del modello.
        """
        num_iter = int(self.URM_train.shape[0] / self.batch_size)
        gen_loss = self.model.cdl_estimate(self.URM_train, num_iter)
        self.model.m_theta[:] = self.model.transform(self.URM_train)
        likelihood = self.model.pmf_estimate(self._train_users, self._train_items, None, None, self._params)
        loss = -likelihood + 0.5 * gen_loss * self.URM_train.shape[0] * self._params.lambda_r

        self.USER_factors = self.model.m_U.copy()
        self.ITEM_factors = self.model.m_V.copy()
        logging.info(
            f"[#epoch={currentEpoch}], loss={loss:.5f}, neg_likelihood={-likelihood:.5f}, gen_loss={gen_loss:.5f}")

    def save_model(self, folder_path, file_name=None):
        """
        Salva il modello addestrato.
        """
        file_name = file_name or self.RECOMMENDER_NAME
        self.model.save_weights(f"{folder_path}{file_name}_weights", overwrite=True)

        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, f"{folder_path}{file_name}_session")

        data_dict_to_save = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "num_factors": self.num_factors,
            "learning_rate": self.learning_rate
        }

        DataIO(folder_path=folder_path).save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

    def load_model(self, folder_path, file_name=None):
        """
        Carica il modello da file.
        """
        file_name = file_name or self.RECOMMENDER_NAME
        data_dict = DataIO(folder_path=folder_path).load_data(file_name=file_name)

        for key, value in data_dict.items():
            setattr(self, key, value)

        self._init_model()
        self.model.load_weights(f"{folder_path}{file_name}_weights")

        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, f"{folder_path}{file_name}_session")

    def _update_best_model(self):
        """
        Salva il miglior modello finora ottenuto.
        """
        self.save_model(self.temp_file_folder, file_name="_best_model")
