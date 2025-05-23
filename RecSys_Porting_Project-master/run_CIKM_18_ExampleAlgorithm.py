#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
#sys.path.append('c:/Users/laral/Desktop/Progetto RS/RS/RecSys_Porting_Project-master')
#sys.path.append('/Users/michelebalena/Desktop/Progetto RS/RS/RecSys_Porting_Project-master')


from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from Recommenders.Recommender_import_list import *
from Utils.ResultFolderLoader import ResultFolderLoader

from functools import partial
import numpy as np
import os, traceback, argparse, multiprocessing

from Conferences.CIKM.ExampleAlgorithm_our_interface.ExampleDatasetProvided.CiteulikeReader import CiteulikeReader
from Conferences.CIKM.ExampleAlgorithm_our_interface.ExampleDatasetPublic.Movielens20MReader import Movielens20MReader

from Conferences.CIKM.ExampleAlgorithm_our_interface.Example_RecommenderWrapper import Example_RecommenderWrapper



from Evaluation.Evaluator import EvaluatorHoldout
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from HyperparameterTuning.hyperparameter_space_library import ExperimentConfiguration
from Utils.RecommenderInstanceIterator import RecommenderConfigurationTupleIterator
from HyperparameterTuning.functions_for_parallel_model import _unpack_tuple_and_search




def read_data_split_and_search(dataset_name,
                               flag_baselines_tune=False,
                               flag_model_default=False,
                               flag_print_results=False):
    

    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)
    data_folder_path = result_folder_path + "data/"
    model_folder_path = result_folder_path + "data/data_split/"

    # TODO: Replace with dataset name and relative DataReader
    #  The two datareaders correspond to two examples, CiteULike as an example of dataset provided in the original repository of the paper you are attempting to reproduce
    #  while Movielens20M as a dataset not provided in the original repository but publicly available, in that case one use one of the data readers
    #  already available in this repository

    if dataset_name == "citeulike":
        dataset = CiteulikeReader(data_folder_path)

    elif dataset_name == "movielens20m":
        dataset = Movielens20MReader(data_folder_path)

    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return


    print ('Current dataset is: {}'.format(dataset_name))

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    ICM_DICT = dataset.ICM_DICT
    UCM_DICT = dataset.UCM_DICT

    URM_train_last_test = URM_train + URM_validation

    # Ensure IMPLICIT data and disjoint test-train split
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])


    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)


    # TODO Replace metric to optimize and cutoffs
    metric_to_optimize = 'NDCG'
    cutoff_to_optimize = 10

    # All cutoffs that will be evaluated are listed here
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]
    max_total_time = 14*24*60*60  # 14 days

    n_cases = 50
    n_processes = 3
    resume_from_saved = True


    # TODO Select the evaluation protocol
    # evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=cutoff_list_validation)
    # evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=cutoff_list_test)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list = [cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)





    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    if flag_model_default:

        try:
            # TODO fill this dictionary with the hyperparameters of the algorithm
            article_hyperparameters = {
                                  "batch_size": 512,
                                  "epochs": 300,
                                  "epochs_MFBPR": 500,
                                  "embedding_size":64,
                                  "hidden_size":128,
                                  "negative_sample_per_positive":1,
                                  "negative_instances_per_positive":4,
                                  "regularization_users_items":0.01,
                                  "regularization_weights":10,
                                  "regularization_filter_weights":1,
                                  "learning_rate_embeddings":0.05,
                                  "learning_rate_CNN":0.05,
                                  "channel_size":[32, 32, 32, 32, 32, 32],
                                  "dropout":0.0,
                                  "epoch_verbose":1,
                                  }

            # Do not modify earlystopping
            earlystopping_hyperparameters = {"validation_every_n": 5,
                                        "stop_on_validation": True,
                                        "lower_validations_allowed": 5,
                                        "evaluator_object": evaluator_validation,
                                        "validation_metric": metric_to_optimize,
                                        }



            # This is a simple version of the tuning code that is reported below and uses SearchSingleCase
            # You may use this for a simpler testing
            # recommender_instance = Example_RecommenderWrapper(URM_train)
            #
            # recommender_instance.fit(**article_hyperparameters,
            #                          **earlystopping_hyperparameters)
            #
            # evaluator_test.evaluateRecommender(recommender_instance)


            # Fit the DL model, select the optimal number of epochs and save the result
            hyperparameterSearch = SearchSingleCase(Example_RecommenderWrapper,
                                                   evaluator_validation=evaluator_validation,
                                                   evaluator_test=evaluator_test)

            # Specify which attributes are needed. In this case the constructor only required the URM train,
            # no additional fit arguments are required (besides those that are listed previously in the hyperparameters dictionary)
            # and the hyperparameters required by the earlystopping are provided separately.
            recommender_input_args = SearchInputRecommenderArgs(
                                                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                                FIT_KEYWORD_ARGS={},
                                                EARLYSTOPPING_KEYWORD_ARGS = earlystopping_hyperparameters)

            # Create the attributes needed to fit the last model on the union of training and validation data
            # This model will be fit with the optimal hyperparameters found and then will be evaluated on the test data
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test

            hyperparameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values={},
                                   metric_to_optimize = metric_to_optimize,
                                   cutoff_to_optimize = cutoff_to_optimize,
                                   output_folder_path = model_folder_path,
                                   output_file_name_root = Example_RecommenderWrapper.RECOMMENDER_NAME,
                                   resume_from_saved = resume_from_saved,
                                   save_model = "best",
                                   evaluate_on_test = "best",
                                   )



        except Exception as e:

            print("On recommender {} Exception {}".format(Example_RecommenderWrapper, str(e)))
            traceback.print_exc()



    ################################################################################################
    ######
    ######      BASELINE ALGORITHMS - Nothing should be modified below this point
    ######

    if flag_baselines_tune:


        recommender_class_list = [
            Random,
            TopPop,
            GlobalEffects,
            # SLIMElasticNetRecommender,
            # UserKNNCFRecommender,
            # MatrixFactorization_BPR_Cython,
            # IALSRecommender,
            # MatrixFactorization_FunkSVD_Cython,
            # EASE_R_Recommender,
            ItemKNNCFRecommender,
            # P3alphaRecommender,
            # SLIM_BPR_Cython,
            # RP3betaRecommender,
            PureSVDRecommender,
            # NMFRecommender,
            # UserKNNCBFRecommender,
            # ItemKNNCBFRecommender,
            # UserKNN_CFCBF_Hybrid_Recommender,
            # ItemKNN_CFCBF_Hybrid_Recommender,
            # LightFMCFRecommender,
            # LightFMUserHybridRecommender,
            # LightFMItemHybridRecommender,
            # MultVAERecommender,
            ]


        # This object contains the data and the information related to the experimental setup
        experiment_configuration = ExperimentConfiguration(
            URM_train=URM_train,
            URM_train_last_test=URM_train_last_test,
            ICM_DICT=ICM_DICT,
            UCM_DICT=UCM_DICT,
            n_cases=n_cases,
            n_random_starts=int(n_cases / 3),
            resume_from_saved=resume_from_saved,
            save_model="last",
            evaluate_on_test="best",
            evaluator_validation=evaluator_validation,
            KNN_similarity_to_report_list=KNN_similarity_to_report_list,
            evaluator_test=evaluator_test,
            max_total_time=max_total_time,
            evaluator_validation_earlystopping=evaluator_validation_earlystopping,
            metric_to_optimize=metric_to_optimize,
            cutoff_to_optimize=cutoff_to_optimize,
            n_processes=n_processes,
        )

        # This iterator creates the cartesian product of all the recommender/URM/ICM-UCM/KNN similarity etc...
        configuration_iterator = RecommenderConfigurationTupleIterator(recommender_class_list = recommender_class_list,
                                                                       KNN_similarity_list = experiment_configuration.KNN_similarity_to_report_list,
                                                                       ICM_name_list = experiment_configuration.ICM_DICT.keys(),
                                                                       UCM_name_list = experiment_configuration.UCM_DICT.keys(),
                                                                       )

        # This function is a helper function needed for parallelization
        _unpack_tuple_and_search_partial = partial(_unpack_tuple_and_search,
                                                   experiment_configuration = experiment_configuration,
                                                   output_folder_path = model_folder_path)

        # This runs the hyperparameter optimization, either parallel or sequentially
        if experiment_configuration.n_processes is not None:
            pool = multiprocessing.Pool(processes=experiment_configuration.n_processes, maxtasksperchild=1)
            pool.map(_unpack_tuple_and_search_partial, configuration_iterator, chunksize=1)

            pool.close()
            pool.join()

        else:
            for model_case_tuple in configuration_iterator:
                _unpack_tuple_and_search_partial(model_case_tuple)




    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)

        result_loader = ResultFolderLoader(model_folder_path,
                                           base_algorithm_list = None,
                                           other_algorithm_list = None,
                                           KNN_similarity_list = KNN_similarity_to_report_list,
                                           ICM_names_list = dataset.ICM_DICT.keys(),
                                           UCM_names_list = dataset.UCM_DICT.keys(),
                                           )

        result_loader.generate_latex_results(result_folder_path + "{}_latex_results.txt".format("accuracy_metrics"),
                                           metrics_list = ['RECALL', 'PRECISION', 'MAP', 'NDCG'],
                                           cutoffs_list = [cutoff_to_optimize],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(result_folder_path + "{}_latex_results.txt".format("beyond_accuracy_metrics"),
                                           metrics_list = ["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = cutoff_list,
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(result_folder_path + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)



if __name__ == '__main__':

    # TODO: Replace with algorithm and conference name
    ALGORITHM_NAME = "EXAMPLE"
    CONFERENCE_NAME = "prova"


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type= bool, default = False)
    parser.add_argument('-m', '--model_default',        help="Train the model with the hyperparameters described in the paper", type= bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type= bool, default = True)


    input_flags = parser.parse_args()
    print(input_flags)

    # Reporting only the cosine similarity is enough
    KNN_similarity_to_report_list = ["cosine"]#, "dice", "jaccard", "asymmetric", "tversky"]


    # TODO: Replace with dataset names, for a runnable example of this pipeline use only movielens20m
    #dataset_list = ["citeulike", "movielens20m"]
    dataset_list = ["movielens20m"]

    for dataset_name in dataset_list:
        read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = input_flags.baseline_tune,
                                   flag_model_default = input_flags.model_default,
                                   flag_print_results = input_flags.print_results,
                                   )

