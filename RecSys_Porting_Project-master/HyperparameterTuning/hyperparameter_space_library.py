#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/07/2023

@author: Maurizio Ferrari Dacrema
"""

######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

# KNN
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.LightGCNRecommender import LightGCNRecommender
from Recommenders.GraphBased.INMORecommender import INMORecommender
from Recommenders.GraphBased.GraphFilterCFRecommender import GraphFilterCF_W_Recommender, GraphFilterCFRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

# KNN machine learning
from Recommenders.SLIM.NegHOSLIM import NegHOSLIMRecommender, NegHOSLIMElasticNetRecommender, NegHOSLIMLSQR
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender

# Matrix Factorization
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_WARP_Cython, \
    MatrixFactorization_SVDpp_Cython, MatrixFactorization_AsySVD_Cython

from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch_OptimizerMask as MultVAERecommender_PyTorch
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender

######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender



######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMItemHybridRecommender, LightFMUserHybridRecommender
from Recommenders.FeatureWeighting.Cython.CFW_D_Similarity_Cython import CFW_D_Similarity_Cython
from Recommenders.FeatureWeighting.Cython.CFW_DVV_Similarity_Cython import CFW_DVV_Similarity_Cython
from Recommenders.FeatureWeighting.Cython.FBSM_Rating_Cython import FBSM_Rating_Cython

######################################################################
from skopt.space import Real, Integer, Categorical

import traceback, os
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

######################################################################


class ExperimentConfiguration(object):
    """
        :param recommender_class:   Class of the recommender object to optimize, it must be a BaseRecommender type
        :param URM_train:           Sparse matrix containing the URM training data
        :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
        :param n_cases:             Number of hyperparameter sets to explore
        :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
        :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
        :param save_model:          ["no", "best", "last"] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
        :param evaluate_on_test:    ["all", "best", "last", "no"] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
        :param max_total_time:    [None or int] if set stops the hyperparameter optimization when the time in seconds for training and validation exceeds the threshold
        :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
        :param evaluator_validation_earlystopping:   Evaluator object to be used for the earlystopping of ML algorithms, can be the same of evaluator_validation
        :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
        :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
        :param cutoff_to_optimize:  Integer with the recommendation list length to be optimized as contained in the output of the evaluator objects
        :param output_folder_path:  Folder in which to save the output files
        :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
        :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
        :param allow_bias_URM:      Boolean value, if True it enables the use of bias to shift the values of the URM
        :param allow_dropout_MF:    Boolean value, if True it enables the use of dropout on the latent factors of MF algorithms
        :param similarity_type:     String with the similarity heuristics to be used for the KNNs
    """
    def __init__(self,
                 URM_train=None,
                 URM_train_last_test=None,
                 ICM_DICT=None,
                 UCM_DICT=None,
                 n_cases=None,
                 n_random_starts=None,
                 resume_from_saved=None,
                 save_model=None,
                 evaluate_on_test=None,
                 evaluator_validation=None,
                 KNN_similarity_to_report_list=None,
                 evaluator_test=None,
                 max_total_time=None,
                 evaluator_validation_earlystopping=None,
                 metric_to_optimize=None,
                 cutoff_to_optimize=None,
                 n_processes=None,
                 ):
        super(ExperimentConfiguration, self).__init__()

        self.URM_train = URM_train
        self.URM_train_last_test = URM_train_last_test
        self.ICM_DICT = ICM_DICT
        self.UCM_DICT = UCM_DICT
        self.n_cases = n_cases
        self.n_random_starts = n_random_starts
        self.resume_from_saved = resume_from_saved
        self.save_model = save_model
        self.evaluate_on_test = evaluate_on_test
        self.evaluator_validation = evaluator_validation
        self.KNN_similarity_to_report_list = KNN_similarity_to_report_list
        self.evaluator_test = evaluator_test
        self.max_total_time = max_total_time
        self.evaluator_validation_earlystopping = evaluator_validation_earlystopping
        self.metric_to_optimize = metric_to_optimize
        self.cutoff_to_optimize = cutoff_to_optimize
        self.n_processes = n_processes



def getHyperparameterSpace(recommender_class, experiment_configuration,
                           ICM_UCM_object, similarity_type,
                           allow_weighting = True, allow_bias_ICM = False, allow_bias_URM = False, allow_dropout_MF = False,
                           ):
    """
    This function returns the hyperparameter space

    :param recommender_class:   Class of the recommender object to optimize, it must be a BaseRecommender type
    :param experiment_configuration:    Object of type ExperimentConfiguration
    :param ICM_UCM_object:      Sparse matrix containing the ICM or UCM training data
    :param output_folder_path:  Folder in which to save the output files
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param allow_bias_ICM:      Boolean value, if True it enables the use of bias to shift the values of the ICM
    :param similarity_type:     String with the similarity heuristics to be used for the KNNs
    """


    URM_train = experiment_configuration.URM_train
    URM_train_last_test = experiment_configuration.URM_train_last_test
    evaluator_validation_earlystopping = experiment_configuration.evaluator_validation_earlystopping
    metric_to_optimize = experiment_configuration.metric_to_optimize


    URM_train = URM_train.copy()
    n_users, n_items = URM_train.shape

    if ICM_UCM_object is not None:
        ICM_UCM_object = ICM_UCM_object.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()


    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation_earlystopping,
                              "lower_validations_allowed": 5,
                              "validation_metric": metric_to_optimize,
                              }

    # try:

    if recommender_class in [TopPop, GlobalEffects, Random]:
        """
        TopPop, GlobalEffects and Random have no hyperparameters therefore only one evaluation is needed
        """

        hyperparameters_range_dictionary = {}

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )

    ##########################################################################################################

    elif recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender,
                             ItemKNNCBFRecommender, UserKNNCBFRecommender,
                             ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender]:

        hyperparameters_range_dictionary = {
            "topK": Integer(5, min(1000, n_items)),
            "shrink": Integer(0, 1000),
            "similarity": Categorical([similarity_type]),
            "normalize": Categorical([True, False]),
        }

        if similarity_type == "asymmetric":
            hyperparameters_range_dictionary["asymmetric_alpha"] = Real(low=0, high=2, prior='uniform')
            hyperparameters_range_dictionary["normalize"] = Categorical([True])

        elif similarity_type == "tversky":
            hyperparameters_range_dictionary["tversky_alpha"] = Real(low=0, high=2, prior='uniform')
            hyperparameters_range_dictionary["tversky_beta"] = Real(low=0, high=2, prior='uniform')
            hyperparameters_range_dictionary["normalize"] = Categorical([True])

        elif similarity_type == "euclidean":
            hyperparameters_range_dictionary["normalize"] = Categorical([True, False])
            hyperparameters_range_dictionary["normalize_avg_row"] = Categorical([True, False])
            hyperparameters_range_dictionary["similarity_from_distance_mode"] = Categorical(["lin", "log", "exp"])


        if recommender_class is ItemKNN_CFCBF_Hybrid_Recommender:
            hyperparameters_range_dictionary["ICM_weight"] = Real(low=1e-2, high=1e2, prior='log-uniform')

        elif recommender_class is UserKNN_CFCBF_Hybrid_Recommender:
            hyperparameters_range_dictionary["UCM_weight"] = Real(low=1e-2, high=1e2, prior='log-uniform')


        is_set_similarity = similarity_type in ["tversky", "dice", "jaccard", "tanimoto"]

        if not is_set_similarity:

            if allow_weighting:
                hyperparameters_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])

            if allow_bias_ICM:
                hyperparameters_range_dictionary["ICM_bias"] = Real(low=1e-2, high=1e+3, prior='log-uniform')

            if allow_bias_URM:
                hyperparameters_range_dictionary["URM_bias"] = Real(low=1e-2, high=1e+3, prior='log-uniform')


        if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {},
            )

        else:
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_UCM_object],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {},
            )


    elif recommender_class in [LightFMItemHybridRecommender, LightFMUserHybridRecommender]:

        hyperparameters_range_dictionary = {
            "epochs": Categorical([300]),
            "n_components": Integer(1, 200),
            "loss": Categorical(['bpr', 'warp', 'warp-kos']),
            "sgd_mode": Categorical(['adagrad', 'adadelta']),
            "learning_rate": Real(low=1e-6, high=1e-1, prior='log-uniform'),
            "item_alpha": Real(low=1e-5, high=1e-2, prior='log-uniform'),
            "user_alpha": Real(low=1e-5, high=1e-2, prior='log-uniform'),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_UCM_object],
            CONSTRUCTOR_KEYWORD_ARGS={},
            FIT_POSITIONAL_ARGS=[],
            FIT_KEYWORD_ARGS={},
            EARLYSTOPPING_KEYWORD_ARGS=earlystopping_keywargs,
        )

   ##########################################################################################################

    elif recommender_class is P3alphaRecommender:

        hyperparameters_range_dictionary = {
            "topK": Integer(5, min(1000, n_items)),
            "alpha": Real(low = 0, high = 2, prior = 'uniform'),
            "normalize_similarity": Categorical([True, False]),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )


    ##########################################################################################################

    elif recommender_class is RP3betaRecommender:

        hyperparameters_range_dictionary = {
            "topK": Integer(5, min(1000, n_items)),
            "alpha": Real(low = 0, high = 2, prior = 'uniform'),
            "beta": Real(low = 0, high = 2, prior = 'uniform'),
            "normalize_similarity": Categorical([True, False]),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )


    ##########################################################################################################

    elif recommender_class is MatrixFactorization_SVDpp_Cython:

        hyperparameters_range_dictionary = {
            "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
            "epochs": Categorical([500]),
            "use_bias": Categorical([True, False]),
            "batch_size": Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "num_factors": Integer(1, 200),
            "item_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "user_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "learning_rate": Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
            "negative_interactions_quota": Real(low = 0.0, high = 0.5, prior = 'uniform'),
        }

        if allow_dropout_MF:
            hyperparameters_range_dictionary["dropout_quota"] = Real(low = 0.01, high = 0.7, prior = 'uniform')

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )

    ##########################################################################################################

    elif recommender_class is MatrixFactorization_AsySVD_Cython:

        hyperparameters_range_dictionary = {
            "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
            "epochs": Categorical([500]),
            "use_bias": Categorical([True, False]),
            "batch_size": Categorical([1]),
            "num_factors": Integer(1, 200),
            "item_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "user_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "learning_rate": Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
            "negative_interactions_quota": Real(low = 0.0, high = 0.5, prior = 'uniform'),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )

    ##########################################################################################################

    elif recommender_class is MatrixFactorization_BPR_Cython:

        hyperparameters_range_dictionary = {
            "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
            "epochs": Categorical([1500]),
            "num_factors": Integer(1, 200),
            "batch_size": Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "positive_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "negative_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "learning_rate": Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
        }

        if allow_dropout_MF:
            hyperparameters_range_dictionary["dropout_quota"] = Real(low = 0.01, high = 0.7, prior = 'uniform')

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {"positive_threshold_BPR": None},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )

    ##########################################################################################################

    elif recommender_class is MatrixFactorization_WARP_Cython:

        hyperparameters_range_dictionary = {
            "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
            "epochs": Categorical([1500]),
            "num_factors": Integer(1, 200),
            "batch_size": Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "WARP_neg_item_attempts": Categorical([5, 10, 15, 20]),
            "positive_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "negative_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "learning_rate": Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
        }

        if allow_dropout_MF:
            hyperparameters_range_dictionary["dropout_quota"] = Real(low = 0.01, high = 0.7, prior = 'uniform')

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {"positive_threshold_BPR": None},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )

    ##########################################################################################################

    elif recommender_class is IALSRecommender:

        hyperparameters_range_dictionary = {
            "num_factors": Integer(1, 200),
            "epochs": Categorical([300]),
            "confidence_scaling": Categorical(["linear", "log"]),
            "alpha": Real(low = 1e-3, high = 50.0, prior = 'log-uniform'),
            "epsilon": Real(low = 1e-3, high = 10.0, prior = 'log-uniform'),
            "reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )


    ##########################################################################################################

    elif recommender_class is PureSVDRecommender:

        hyperparameters_range_dictionary = {
            "num_factors": Integer(1, min(350, n_items-1)),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )


    ##########################################################################################################

    elif recommender_class is PureSVDItemRecommender:

        hyperparameters_range_dictionary = {
            "num_factors": Integer(1, min(350, n_items-1)),
            "topK": Integer(5, min(1000, n_items)),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )


    ##########################################################################################################

    elif recommender_class is NMFRecommender:

        hyperparameters_range_dictionary = {
            "num_factors": Integer(1, min(350, n_items-1)),
            "solver_beta_loss": Categorical(["coordinate_descent:frobenius", "multiplicative_update:frobenius", "multiplicative_update:kullback-leibler"]),
            "init_type": Categorical(["random", "nndsvda"]),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )


    #########################################################################################################

    elif recommender_class is SLIM_BPR_Cython:

        hyperparameters_range_dictionary = {
            "topK": Integer(5, min(1000, n_items)),
            "epochs": Categorical([1500]),
            "symmetric": Categorical([True, False]),
            "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
            "lambda_i": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "lambda_j": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "learning_rate": Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {"positive_threshold_BPR": None,
                                'train_with_sparse_weights': False,
                                'allow_train_with_sparse_weights': False},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )



    ##########################################################################################################

    elif recommender_class is SLIMElasticNetRecommender or recommender_class is MultiThreadSLIM_SLIMElasticNetRecommender:

        hyperparameters_range_dictionary = {
            "topK": Integer(5, min(1000, n_items)),
            "l1_ratio": Real(low = 1e-5, high = 1.0, prior = 'log-uniform'),
            "alpha": Real(low = 1e-3, high = 1.0, prior = 'uniform'),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )


    #########################################################################################################

    elif recommender_class is EASE_R_Recommender:

        hyperparameters_range_dictionary = {
            "topK": Categorical([None]),
            "normalize_matrix": Categorical([False]),
            "l2_norm": Real(low = 1e0, high = 1e7, prior = 'log-uniform'),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )


    #########################################################################################################

    elif recommender_class is NegHOSLIMRecommender:

        hyperparameters_range_dictionary = {
            # "feature_pairs_threshold": Real(low = 1e0, high = 1e7, prior = 'log-uniform'),
            "feature_pairs_n": Integer(1, 1000),
            "lambdaBB": Real(low = 1e0, high = 1e7, prior = 'log-uniform'),
            "lambdaCC": Real(low = 1e0, high = 1e7, prior = 'log-uniform'),
            "rho": Real(low = 1e0, high = 1e7, prior = 'log-uniform'),
            "epochs": Categorical([300]),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )


    #########################################################################################################

    elif recommender_class is NegHOSLIMElasticNetRecommender:

        hyperparameters_range_dictionary = {
            # "feature_pairs_threshold": Real(low = 1e0, high = 1e7, prior = 'log-uniform'),
            "feature_pairs_n": Integer(1, 1000),
            "topK": Integer(5, min(1000, n_items)),
            "positive_only_weights": Categorical([False]),
            "l1_ratio": Real(low = 1e-5, high = 1.0, prior = 'log-uniform'),
            "alpha": Real(low = 1e-3, high = 1.0, prior = 'uniform'),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )

    #########################################################################################################

    elif recommender_class is NegHOSLIMLSQR:

        hyperparameters_range_dictionary = {
            # "feature_pairs_threshold": Real(low = 1e0, high = 1e7, prior = 'log-uniform'),
            "feature_pairs_n": Integer(1, 1000),
            "topK": Integer(5, min(1000, n_items)),
            "damp": Real(low = 1e-3, high = 1.0, prior = 'uniform'),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )

    #########################################################################################################

    elif recommender_class is LightFMCFRecommender:

        hyperparameters_range_dictionary = {
            "epochs": Categorical([300]),
            "n_components": Integer(1, 200),
            "loss": Categorical(['bpr', 'warp', 'warp-kos']),
            "sgd_mode": Categorical(['adagrad', 'adadelta']),
            "learning_rate": Real(low = 1e-6, high = 1e-1, prior = 'log-uniform'),
            "item_alpha": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            "user_alpha": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )

    #########################################################################################################

    elif recommender_class is MultVAERecommender:

        hyperparameters_range_dictionary = {
            "epochs": Categorical([500]),
            "learning_rate": Real(low=1e-6, high=1e-2, prior="log-uniform"),
            "l2_reg": Real(low=1e-6, high=1e-2, prior="log-uniform"),
            "dropout": Real(low=0., high=0.8, prior="uniform"),
            "total_anneal_steps": Integer(100000, 600000),
            "anneal_cap": Real(low=0., high=0.6, prior="uniform"),
            "batch_size": Categorical([128, 256, 512, 1024]),

            "encoding_size": Integer(1, min(512, n_items-1)),
            "next_layer_size_multiplier": Integer(2, 10),
            "max_n_hidden_layers": Integer(1, 4),

            # Constrain the model to a maximum number of parameters so that its size does not exceed 1.45 GB
            # Estimate size by considering each parameter uses float32
            "max_parameters": Categorical([1.45*1e9*8/32]),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )




    #########################################################################################################

    elif recommender_class is MultVAERecommender_PyTorch:

        hyperparameters_range_dictionary = {
            "epochs": Categorical([500]),
            "learning_rate": Real(low=1e-6, high=1e-2, prior="log-uniform"),
            "sgd_mode": Categorical(["sgd", "adagrad", "adam", "rmsprop"]),
            "l2_reg": Real(low=1e-6, high=1e-2, prior="log-uniform"),
            "dropout": Real(low=0., high=0.8, prior="uniform"),
            "total_anneal_steps": Integer(100000, 600000),
            "anneal_cap": Real(low=0., high=0.6, prior="uniform"),
            "batch_size": Categorical([128, 256, 512, 1024]),

            "encoding_size": Integer(1, min(512, n_items-1)),
            "next_layer_size_multiplier": Integer(2, 10),
            "max_n_hidden_layers": Integer(1, 4),

            # Constrain the model to a maximum number of parameters so that its size does not exceed 1.45 GB
            # Estimate size by considering each parameter uses float32
            "max_parameters": Categorical([1.45*1e9*8/32]),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {"verbose": False, "use_gpu": True},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )

    #########################################################################################################

    elif recommender_class is LightGCNRecommender:

        hyperparameters_range_dictionary = {
            "epochs": Categorical([1000]),
            "batch_size": Categorical([256, 512, 1024, 2048, 4096]),
            "learning_rate": Real(low=1e-6, high=1e-1, prior="log-uniform"),

            "embedding_size": Integer(2, 350),
            "sgd_mode": Categorical(["sgd", "adagrad", "adam", "rmsprop"]),

            "GNN_layers_K": Integer(1, 6),  # The original paper limits it to 4
            "l2_reg": Real(low=0.1, high=0.9, prior="uniform"),
            "dropout_rate": Real(low=1e-6, high=1e-1, prior="log-uniform"),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {"use_gpu": True, "verbose":False},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )


    #########################################################################################################

    elif recommender_class is INMORecommender:

        hyperparameters_range_dictionary = {
            "epochs": Categorical([1000]),
            "batch_size": Categorical([256, 512, 1024, 2048, 4096]),
            "learning_rate": Real(low=1e-6, high=1e-1, prior="log-uniform"),

            "embedding_size": Integer(2, 350),
            "sgd_mode": Categorical(["adam"]),

            "GNN_layers_K": Integer(1, 6),
            "l2_reg": Real(low=1e-6, high=1e-1, prior="log-uniform"),
            "template_loss_weight": Real(low=1e-4, high=1e-1, prior="log-uniform"),
            "template_node_ranking_metric": Categorical(["degree", "sort", "page_rank"]),
            "dropout_rate": Real(low=0.1, high=0.9, prior="uniform"),
            "template_ratio": Real(low=0.1, high=1.0, prior="uniform"),

            "normalization_decay": Categorical([0.99]),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {"use_gpu": True, "verbose":False},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
        )


    #########################################################################################################


    elif recommender_class is GraphFilterCF_W_Recommender:

        hyperparameters_range_dictionary = {
            "topK": Integer(5, min(5000, n_items-1)),       # This model requires a larger topK
            "alpha": Real(low=1e-3, high=1e3, prior="log-uniform"),
            "num_factors": Integer(1, 350),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )

   ##########################################################################################################

    elif recommender_class is GraphFilterCFRecommender:

        hyperparameters_range_dictionary = {
            "alpha": Real(low=1e-3, high=1e3, prior="log-uniform"),
            "num_factors": Integer(1, 350),
        }

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {},
            EARLYSTOPPING_KEYWORD_ARGS = {},
        )

   ##########################################################################################################
    #
    # elif recommender_class is FBSM_Rating_Cython:
    #     hyperparameters_range_dictionary = {
    #         "topK": Categorical([300]),
    #         "n_factors": Integer(1, 5),
    #
    #         "learning_rate": Real(low=1e-5, high=1e-2, prior='log-uniform'),
    #         "sgd_mode": Categorical(["adam"]),
    #         "l2_reg_D": Real(low=1e-6, high=1e1, prior='log-uniform'),
    #         "l2_reg_V": Real(low=1e-6, high=1e1, prior='log-uniform'),
    #         "epochs": Categorical([300]),
    #     }
    #
    #     recommender_input_args = SearchInputRecommenderArgs(
    #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_object],
    #         CONSTRUCTOR_KEYWORD_ARGS={},
    #         FIT_POSITIONAL_ARGS=[],
    #         FIT_KEYWORD_ARGS={"validation_every_n": 5,
    #                           "stop_on_validation": True,
    #                           "evaluator_object": evaluator_validation_earlystopping,
    #                           "lower_validations_allowed": 10,
    #                           "validation_metric": metric_to_optimize}
    #     )
    #
    # elif recommender_class is CFW_D_Similarity_Cython:
    #     hyperparameters_range_dictionary = {
    #         "topK": Categorical([300]),
    #
    #         "learning_rate": Real(low=1e-5, high=1e-2, prior='log-uniform'),
    #         "sgd_mode": Categorical(["adam"]),
    #         "l1_reg": Real(low=1e-3, high=1e-2, prior='log-uniform'),
    #         "l2_reg": Real(low=1e-3, high=1e-1, prior='log-uniform'),
    #         "epochs": Categorical([50]),
    #
    #         "init_type": Categorical(["one", "random"]),
    #         "add_zeros_quota": Real(low=0.50, high=1.0, prior='uniform'),
    #         "positive_only_weights": Categorical([True, False]),
    #         "normalize_similarity": Categorical([True]),
    #
    #         "use_dropout": Categorical([True]),
    #         "dropout_perc": Real(low=0.30, high=0.8, prior='uniform'),
    #     }
    #
    #     recommender_input_args = SearchInputRecommenderArgs(
    #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_object, W_train],
    #         CONSTRUCTOR_KEYWORD_ARGS={},
    #         FIT_POSITIONAL_ARGS=[],
    #         FIT_KEYWORD_ARGS={"precompute_common_features": False,  # Reduces memory requirements
    #                           "validation_every_n": 5,
    #                           "stop_on_validation": True,
    #                           "evaluator_object": evaluator_validation_earlystopping,
    #                           "lower_validations_allowed": 10,
    #                           "validation_metric": metric_to_optimize}
    #     )
    #
    # elif recommender_class is CFW_DVV_Similarity_Cython:
    #
    #     hyperparameters_range_dictionary = {
    #         "topK": Categorical([300]),
    #         "n_factors": Integer(1, 2),
    #
    #         "learning_rate": Real(low=1e-5, high=1e-3, prior='log-uniform'),
    #         "sgd_mode": Categorical(["adam"]),
    #         "l2_reg_D": Real(low=1e-6, high=1e1, prior='log-uniform'),
    #         "l2_reg_V": Real(low=1e-6, high=1e1, prior='log-uniform'),
    #         "epochs": Categorical([100]),
    #
    #         "add_zeros_quota": Real(low=0.50, high=1.0, prior='uniform'),
    #     }
    #
    #     recommender_input_args = SearchInputRecommenderArgs(
    #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_object, W_train],
    #         CONSTRUCTOR_KEYWORD_ARGS={},
    #         FIT_POSITIONAL_ARGS=[],
    #         FIT_KEYWORD_ARGS={"precompute_common_features": False,  # Reduces memory requirements
    #                           "validation_every_n": 5,
    #                           "stop_on_validation": True,
    #                           "evaluator_object": evaluator_validation_earlystopping,
    #                           "lower_validations_allowed": 10,
    #                           "validation_metric": metric_to_optimize}
    #     )

    else:
        raise NotImplementedError("Recommender class not recognized as having a hyperparameter space.")



    if URM_train_last_test is not None:
        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
    else:
        recommender_input_args_last_test = None

    return hyperparameters_range_dictionary, recommender_input_args, recommender_input_args_last_test

    #
    # except Exception as e:
    #
    #     print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #     traceback.print_exc()
    #
    #     error_file = open(output_folder_path + "ErrorLog.txt", "a")
    #     error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
    #     error_file.close()









def runHyperparameterSearch(recommender_class, experiment_configuration, output_folder_path,
                          ICM_UCM_object = None, ICM_UCM_name = None, similarity_type = None,
                          allow_weighting = True, allow_bias_URM = False, allow_bias_ICM = False, allow_dropout_MF = False):

    """
    This function performs the hyperparameter optimization for a recommender

    :param recommender_class:   Class of the recommender object to optimize, it must be a BaseRecommender type
    :param experiment_configuration:    Object of type ExperimentConfiguration
    :param output_folder_path:  Path of the folder in which the models will be saved
    :param ICM_UCM_object:          Sparse matrix containing the ICM or UCM training data
    :param ICM_UCM_name:            String with the name of the ICM or UCM
    :param similarity_type:     String with the similarity heuristics to be used for the KNNs
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param allow_bias_URM:      Boolean value, if True it enables the use of bias to shift the values of the URM
    :param allow_bias_ICM:      Boolean value, if True it enables the use of bias to shift the values of the ICM
    :param allow_dropout_MF:    Boolean value, if True it enables the use of dropout on the latent factors of MF algorithms
    """



    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    hyperparameters_range_dictionary, recommender_input_args, recommender_input_args_last_test = getHyperparameterSpace(recommender_class,
                           experiment_configuration,
                           ICM_UCM_object, similarity_type,
                           allow_weighting = allow_weighting,
                           allow_bias_ICM = allow_bias_ICM,
                           allow_bias_URM = allow_bias_URM,
                           allow_dropout_MF = allow_dropout_MF,
                           )

    try:

        recommender_name_full = "{}{}{}".format(recommender_class.RECOMMENDER_NAME,
                                                "_{}".format(ICM_UCM_name) if ICM_UCM_name is not None else "",
                                                "_{}".format(similarity_type) if similarity_type is not None else "")

        if recommender_class in [TopPop, GlobalEffects, Random]:
            """
            TopPop, GlobalEffects and Random have no hyperparameters therefore only one evaluation is needed
            """

            hyperparameterSearch = SearchSingleCase(recommender_class,
                                                    evaluator_validation = experiment_configuration.evaluator_validation,
                                                    evaluator_test = experiment_configuration.evaluator_test)

            hyperparameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values={},
                                   metric_to_optimize = experiment_configuration.metric_to_optimize,
                                   cutoff_to_optimize = experiment_configuration.cutoff_to_optimize,
                                   output_folder_path = output_folder_path,
                                   output_file_name_root = recommender_name_full,
                                   resume_from_saved = experiment_configuration.resume_from_saved,
                                   save_model = experiment_configuration.save_model,
                                   evaluate_on_test = experiment_configuration.evaluate_on_test,
                                   )

        else:
            hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                                    evaluator_validation = experiment_configuration.evaluator_validation,
                                                    evaluator_test = experiment_configuration.evaluator_test)

            hyperparameterSearch.search(recommender_input_args,
                                   hyperparameter_search_space= hyperparameters_range_dictionary,
                                   n_cases = experiment_configuration.n_cases,
                                   n_random_starts = experiment_configuration.n_random_starts,
                                   resume_from_saved = experiment_configuration.resume_from_saved,
                                   save_model = experiment_configuration.save_model,
                                   evaluate_on_test = experiment_configuration.evaluate_on_test,
                                   max_total_time = experiment_configuration.max_total_time,
                                   output_folder_path = output_folder_path,
                                   output_file_name_root = recommender_name_full,
                                   metric_to_optimize = experiment_configuration.metric_to_optimize,
                                   cutoff_to_optimize = experiment_configuration.cutoff_to_optimize,
                                   recommender_input_args_last_test = recommender_input_args_last_test)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()








def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
    from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
    from Utils.RecommenderInstanceIterator import RecommenderConfigurationTupleIterator
    from Evaluation.Evaluator import EvaluatorHoldout
    from HyperparameterTuning.functions_for_parallel_model import _unpack_tuple_and_search
    import multiprocessing, traceback
    from functools import partial


    dataset_reader = Movielens1MReader()
    output_folder_path = "result_experiments/SKOPT_test/"
    model_folder_path = output_folder_path + "models/"

    dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10])
    dataSplitter.load_data(save_folder_path=output_folder_path + "data/")

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    URM_train_last_test = URM_train + URM_validation


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    recommender_class_list = [
        Random,
        TopPop,
        P3alphaRecommender,
        RP3betaRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        ItemKNNCBFRecommender,
        ItemKNN_CFCBF_Hybrid_Recommender,
        UserKNNCBFRecommender,
        UserKNN_CFCBF_Hybrid_Recommender
    ]

    KNN_similarity_to_report_list = ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean']
    metric_to_optimize = 'NDCG'
    cutoff_to_optimize = 10
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]
    max_total_time = 10*60 # 10 minutes
    n_cases = 50
    n_processes = 5


    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)


    experiment_configuration = ExperimentConfiguration(
        URM_train=URM_train,
        URM_train_last_test=URM_train_last_test,
        ICM_DICT=dataSplitter.get_loaded_ICM_dict(),
        UCM_DICT=dataSplitter.get_loaded_UCM_dict(),
        n_cases=n_cases,
        n_random_starts=int(n_cases / 3),
        resume_from_saved=True,
        save_model="last",
        evaluate_on_test="best",
        evaluator_validation=evaluator_validation,
        KNN_similarity_to_report_list=None,
        evaluator_test=evaluator_test,
        max_total_time=max_total_time,
        evaluator_validation_earlystopping=evaluator_validation_earlystopping,
        metric_to_optimize=metric_to_optimize,
        cutoff_to_optimize=cutoff_to_optimize,
        n_processes=None,
    )

    configuration_iterator = RecommenderConfigurationTupleIterator(recommender_class_list=recommender_class_list,
                                                                   KNN_similarity_list=KNN_similarity_to_report_list,
                                                                   ICM_name_list=experiment_configuration.ICM_DICT.keys(),
                                                                   UCM_name_list=experiment_configuration.UCM_DICT.keys(),
                                                                   )


    _unpack_tuple_and_search_partial = partial(_unpack_tuple_and_search,
                                               n_cases=n_cases,
                                               resume_from_saved=experiment_configuration.resume_from_saved,
                                               max_total_time=experiment_configuration.max_total_time,
                                               metric_to_optimize=experiment_configuration.metric_to_optimize,
                                               cutoff_list=cutoff_list,
                                               cutoff_to_optimize=cutoff_to_optimize)


    if n_processes is None:
        for dataset_model_tuple in configuration_iterator:
            try:
                _unpack_tuple_and_search_partial(dataset_model_tuple)
            except Exception as e:
                print("On dataset {} Exception {}".format(dataset_model_tuple[0], str(e)))
                traceback.print_exc()

    else:
        pool = multiprocessing.Pool(processes=n_processes, maxtasksperchild=1)
        result_list = pool.map(_unpack_tuple_and_search_partial, configuration_iterator, chunksize=1)

        pool.close()
        pool.join()


if __name__ == '__main__':


    read_data_split_and_search()
