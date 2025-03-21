#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
#import tensorflow

from Conferences.RecSysProject.CODIGEM_github import main
from Conferences.RecSysProject.CODIGEM_github.ddgm_model_rs import DDGM
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
            data = data_tr[e_idxlist[start_idx:end_idx]]

            data_tensor = torch.FloatTensor(data.toarray())

            if total_anneal_steps > 0:
                anneal = min(anneal_cap,
                             1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap


            loss, recon_batch = self.model.forward(data_tensor, anneal)




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


        p_dnns = nn.ModuleList([nn.Sequential(nn.Linear(self.D, self.M), nn.PReLU(),
                                              nn.Linear(self.M, self.M), nn.PReLU(),
                                              nn.Linear(self.M, self.M), nn.PReLU(),
                                              nn.Linear(self.M, self.M), nn.PReLU(),
                                              nn.Linear(self.M, self.M), nn.PReLU(),
                                              nn.Linear(self.M, 2 * self.D)) for _ in range(self.T - 1)])

        decoder_net = nn.Sequential(nn.Linear(self.D, self.M), nn.PReLU(),
                                    nn.Linear(self.M, self.M), nn.PReLU(),
                                    nn.Linear(self.M, self.M), nn.PReLU(),
                                    nn.Linear(self.M, self.M), nn.PReLU(),
                                    nn.Linear(self.M, self.M), nn.PReLU(),
                                    nn.Linear(self.M, self.D), nn.Tanh())

        self.model =DDGM(p_dnns, decoder_net, self.beta, self.T, self.D)



    def fit(self,
            M=200,
            epochs=1,
            T=3,
            lr=0.001,
            beta=0.0001,

            # Parametri standard
            temp_file_folder=None,
            **earlystopping_kwargs
            ):

        self.D = self.n_items  # input dimension
        self.M = M  # the number of neurons in scale (s) and translation (t) nets
        self.T = T  # hyperparater to tune
        self.beta = beta  # hyperparater to tune #Beta = 0.0001 is best so far
        self.lr = lr  # learning rate
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




        self._init_model()

        self.optimizer = torch.optim.Adamax(
            [p for p in self.model.parameters() if p.requires_grad == True], lr=self.lr)

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
        # Training and Validation procedure
        #aspettati errore del formato di URM_train
        dg.training(
            model=self.model, optimizer=self.optimizer, training_loader=self.URM_train)










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

