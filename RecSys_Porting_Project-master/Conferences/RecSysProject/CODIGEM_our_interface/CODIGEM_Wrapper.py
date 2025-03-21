#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
#import tensorflow

from Conferences.RecSysProject.CODIGEM_github import main
from Conferences.RecSysProject.CODIGEM_github.main import *

from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.DataIO import DataIO
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping




# TODO replace the recommender class name with the correct one
class CODIGEM_RecommenderWrapper(BaseItemCBFRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    # TODO replace the recommender name with the correct one
    RECOMMENDER_NAME = "CODIGEM_RecommenderWrapper"

    '''def __init__(self, URM_train, ICM_train):
        # TODO remove ICM_train and inheritance from BaseItemCBFRecommender if content features are not needed
        super(CODIGEM_RecommenderWrapper, self).__init__(URM_train, ICM_train)

        # This is used in _compute_item_score
        self._item_indices = np.arange(0, self.n_items, dtype=np.int)'''
    def __init__(self, URM_train):
        # TODO remove ICM_train and inheritance from BaseItemCBFRecommender if content features are not needed
        super(CODIGEM_RecommenderWrapper, self).__init__(URM_train)

        # This is used in _compute_item_score
        self._item_indices = np.arange(0, self.n_items, dtype=np.int)


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # TODO if the model in the end is either a matrix factorization algorithm or an ItemKNN/UserKNN
        #  you can have this class inherit from BaseMatrixFactorization, BaseItemSimilarityMatrixRecommender
        #  or BaseUSerSimilarityMatrixRecommender
        #  in which case you do not have to re-implement this function, you only need to set the
        #  USER_factors, ITEM_factors (see PureSVD) or W_Sparse (see ItemKNN) data structures in the FIT function
        # In order to compute the prediction the model may need a Session. The session is an attribute of this Wrapper.
        # There are two possible scenarios for the creation of the session: at the beginning of the fit function (training phase)
        # or at the end of the fit function (before loading the best model, testing phase)

        # Do not modify this
        # Create the full data structure that will contain the item scores
        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        if items_to_compute is not None:
            item_indices = items_to_compute
        else:
            item_indices = self._item_indices


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            # TODO this predict function should be replaced by whatever code is needed to compute the prediction for a user

            # The prediction requires a list of two arrays user_id, item_id of equal length
            # To compute the recommendations for a single user, we must provide its index as many times as the
            # number of items
            item_score_user = self.model.predict([self._user_ones_vector*user_id, item_indices],
                                                 batch_size=100, verbose=0)

            # Do not modify this
            # Put the predictions in the correct items
            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = item_score_user.ravel()[items_to_compute]
            else:
                item_scores[user_index, :] = item_score_user.ravel()


        return item_scores


    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        #tensorflow.reset_default_graph()

        # TODO Instantiate the model
        # Always clear the default graph if using tehsorflow



        self.model = get_model(num_users = self.n_users,
                          num_items = self.n_items,
                          params=self._params,
                          loss_type='cross-entropy',
                          print_step=10,
                          verbose=False)



    def fit(self,
            epochs=1,
            num_factors=64,
            batch_size=200,
            learning_rate=0.001,

            # Parametri standard
            temp_file_folder=None,
            **earlystopping_kwargs
            ):


        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)


        # TODO replace the following code with what needed to create an instance of the model.
        #  Preferably create an init_model function
        #  If you are using tensorflow before creating the model call tf.reset_default_graph()

        # The following code contains various operations needed by another wrapper

        self.learning_rate = learning_rate
        self.num_factors = num_factors
        self.batch_size = batch_size

    # These are the train instances as a list of lists
        # The following code processed the URM into the data structure the model needs to train
        self._run_epoch(1)
        self._train_users = []

        self.URM_train = sps.csr_matrix(self.URM_train)

        for user_index in range(self.n_users):

            start_pos = self.URM_train.indptr[user_index]
            end_pos = self.URM_train.indptr[user_index +1]

            user_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_users.append(list(user_profile))


        self._train_items = []

        self.URM_train = sps.csc_matrix(self.URM_train)

        for user_index in range(self.n_items):

            start_pos = self.URM_train.indptr[user_index]
            end_pos = self.URM_train.indptr[user_index +1]

            item_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_items.append(list(item_profile))





        self.URM_train = sps.csr_matrix(self.URM_train)



        self._init_model()



        # TODO Close all sessions used for training and open a new one for the "_best_model"
        # close session tensorflow
        #self.sess.close()
        #self.sess = tensorflow.Session()

        ###############################################################################
        ### This is a standard training with early stopping part, most likely you won't need to change it

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Training complete".format(self.RECOMMENDER_NAME))



    def _prepare_model_for_validation(self):
        # TODO Most likely you won't need to change this function
        pass


    def _update_best_model(self):
        # TODO Most likely you won't need to change this function
        self.save_model(self.temp_file_folder, file_name="_best_model")





    def _run_epoch(self, currentEpoch):
        # TODO replace this with the train loop for one epoch of the model
        print(f"Eseguendo il training per l'epoca {currentEpoch}...")
        main.run_training()

        '''n = self.ICM_train.shape[0]

        # for epoch in range(self._params.n_epochs):
        num_iter = int(n / self._params.batch_size)
        # gen_loss = self.cdl_estimate(data_x, params.cdl_max_iter)
        gen_loss = self.model.cdl_estimate(self.ICM_train, num_iter)
        self.model.m_theta[:] = self.model.transform(self.ICM_train)
        likelihood = self.model.pmf_estimate(self._train_users, self._train_items, None, None, self._params)
        loss = -likelihood + 0.5 * gen_loss * n * self._params.lambda_r

        self.USER_factors = self.model.m_U.copy()
        self.ITEM_factors = self.model.m_V.copy()

        logging.info("[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, gen_loss=%.5f" % (
            currentEpoch, loss, -likelihood, gen_loss))'''








    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        # TODO replace this with the Saver required by the model
        #  in this case the neural network will be saved with the _weights suffix, which is rather standard
        self.model.save_weights(folder_path + file_name + "_weights", overwrite=True)

        # TODO Alternativley you may save the tensorflow model with a session
        #saver = tensorflow.train.Saver()
        #saver.save(self.sess, folder_path + file_name + "_session")

        data_dict_to_save = {
            # TODO replace this with the hyperparameters and attribute list you need to re-instantiate
            #  the model when calling the load_model
            "n_users": self.n_users,
            "n_items": self.n_items,
            "mf_dim": self.mf_dim,
            "layers": self.layers,
            "reg_layers": self.reg_layers,
            "reg_mf": self.reg_mf,
        }

        # Do not change this
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")




    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        # Reload the attributes dictionary
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])


        # TODO replace this with what required to re-instantiate the model and load its weights,
        #  Call the init_model function you created before
        self._init_model()
        self.model.load_weights(folder_path + file_name + "_weights")

        # TODO If you are using tensorflow, you may instantiate a new session here
        # TODO reset the default graph to "clean" the tensorflow state
        #tensorflow.reset_default_graph()
        #saver = tensorflow.train.Saver()
        #saver.restore(self.sess, folder_path + file_name + "_session")


        self._print("Loading complete")

