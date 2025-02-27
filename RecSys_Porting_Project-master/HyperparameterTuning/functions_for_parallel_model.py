#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/07/2022

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import os, traceback, multiprocessing
from Evaluation.Evaluator import EvaluatorHoldout
from functools import partial

from Utils.RecommenderInstanceIterator import RecommenderConfigurationTupleIterator
from HyperparameterTuning.hyperparameter_space_library import ExperimentConfiguration

from Data_manager.data_consistency_check import assert_disjoint_matrices, assert_implicit_data

from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
from Utils.ResultFolderLoader import ResultFolderLoader
from Utils.all_dataset_stats_latex_table import all_dataset_stats_latex_table
from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataPostprocessing_K_Cores import DataPostprocessing_K_Cores

from HyperparameterTuning.hyperparameter_space_library import runHyperparameterSearch

def _remove_if_present(object, collection):
    if object in collection:
        collection.remove(object)


def _unpack_tuple_and_search(model_tuple, experiment_configuration, output_folder_path):

    try:

        if model_tuple.ICM_name is not None:
            ICM_UCM_object = experiment_configuration.ICM_DICT[model_tuple.ICM_name]
            ICM_UCM_name = model_tuple.ICM_name
        elif model_tuple.UCM_name is not None:
            ICM_UCM_object = experiment_configuration.UCM_DICT[model_tuple.UCM_name]
            ICM_UCM_name = model_tuple.UCM_name
        else:
            ICM_UCM_object = None
            ICM_UCM_name = None

        runHyperparameterSearch(model_tuple.recommender_class, experiment_configuration,
                                output_folder_path,
                                ICM_UCM_object = ICM_UCM_object,
                                ICM_UCM_name = ICM_UCM_name,
                                similarity_type = model_tuple.KNN_similarity,
                                allow_weighting = True,
                                allow_bias_URM = False,
                                allow_bias_ICM = False,
                                allow_dropout_MF = False)

    except Exception as e:
        print("On Config {} Exception {}".format(model_tuple, str(e)))
        traceback.print_exc()



def _make_data_implicit(dataSplitter):

    dataSplitter.SPLIT_URM_DICT["URM_train"].data = np.ones_like(dataSplitter.SPLIT_URM_DICT["URM_train"].data)
    dataSplitter.SPLIT_URM_DICT["URM_validation"].data = np.ones_like(dataSplitter.SPLIT_URM_DICT["URM_validation"].data)
    dataSplitter.SPLIT_URM_DICT["URM_test"].data = np.ones_like(dataSplitter.SPLIT_URM_DICT["URM_test"].data)

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

    assert_disjoint_matrices([URM_train, URM_validation, URM_test])


def _get_data_split_and_folders(dataset_class, split_type, preprocessing, k_cores = 0):

    dataset_reader = dataset_class()

    if k_cores>0:
        dataset_reader = DataPostprocessing_K_Cores(dataset_reader, k_cores_value=k_cores)

    result_folder_path = "result_experiments/{}/{}/hyperopt_{}/{}/".format("k_{}_cores".format(k_cores) if k_cores>0 else "full",
                                                                           "original",
                                                                           split_type,
                                                                           dataset_reader._get_dataset_name())

    if split_type == "random_holdout_80_10_10":
        dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10], forbid_new_split = True)

    elif split_type == "leave_1_out":
        dataSplitter = DataSplitter_leave_k_out(dataset_reader, k_out_value = 1, use_validation_set = True, leave_random_out = True, forbid_new_split = True)
    else:
        raise ValueError

    data_folder_path = result_folder_path + "data/"
    dataSplitter.load_data(save_folder_path=data_folder_path)

    result_folder_path = "result_experiments/{}/{}/hyperopt_{}/{}/".format("k_{}_cores".format(k_cores) if k_cores>0 else "full",
                                                                           preprocessing,
                                                                           split_type,
                                                                           dataset_reader._get_dataset_name())

    # Save statistics if they do not exist
    if not os.path.isfile(data_folder_path + "item_popularity_plot"):

        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        plot_popularity_bias([URM_train + URM_validation, URM_test],
                             ["Training data", "Test data"],
                             data_folder_path + "item_popularity_plot")

        save_popularity_statistics([URM_train + URM_validation, URM_test],
                                   ["Training data", "Test data"],
                                   data_folder_path + "item_popularity_statistics.tex")

        all_dataset_stats_latex_table(URM_train + URM_validation + URM_test, dataset_reader._get_dataset_name(),
                                      data_folder_path + "dataset_stats.tex")


    if preprocessing == "implicit":
        _make_data_implicit(dataSplitter)

    model_folder_path = result_folder_path + "models/"

    return dataSplitter, result_folder_path, data_folder_path, model_folder_path



def read_data_split_and_search(dataset_class,
                               recommender_class_list,
                               KNN_similarity_to_report_list = [],
                               flag_baselines_tune=False,
                               flag_print_results=False,
                               metric_to_optimize = None,
                               cutoff_to_optimize = None,
                               cutoff_list = None,
                               n_cases = None,
                               max_total_time = None,
                               resume_from_saved = True,
                               split_type = None,
                               preprocessing = None,
                               k_cores = 0,
                               n_processes = 4):

    if preprocessing == "implicit" and dataset_class.IS_IMPLICIT:
        # If preprocessing should be applied but the dataset is already implicit, there is nothing to do
        print("Dataset {} is already implicit, skipping...".format(dataset_class))
        return

    dataSplitter, result_folder_path, data_folder_path, model_folder_path = _get_data_split_and_folders(dataset_class, split_type, preprocessing, k_cores)

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    URM_train_last_test = URM_train + URM_validation

    # Ensure disjoint test-train split
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    if preprocessing == "implicit":
        assert_implicit_data([URM_train, URM_validation, URM_test, URM_train_last_test])

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list = [cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)


    if flag_baselines_tune:

        experiment_configuration = ExperimentConfiguration(
            URM_train=URM_train,
            URM_train_last_test=URM_train_last_test,
            ICM_DICT=dataSplitter.get_loaded_ICM_dict(),
            UCM_DICT=dataSplitter.get_loaded_UCM_dict(),
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

        configuration_iterator = RecommenderConfigurationTupleIterator(recommender_class_list = recommender_class_list,
                                                                       KNN_similarity_list = experiment_configuration.KNN_similarity_to_report_list,
                                                                       ICM_name_list = experiment_configuration.ICM_DICT.keys(),
                                                                       UCM_name_list = experiment_configuration.UCM_DICT.keys(),
                                                                       )

        _unpack_tuple_and_search_partial = partial(_unpack_tuple_and_search,
                                                   experiment_configuration = experiment_configuration,
                                                   output_folder_path = model_folder_path)

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

    if flag_print_results and os.path.exists(model_folder_path):

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)

        result_loader = ResultFolderLoader(model_folder_path,
                                           base_algorithm_list = None,
                                           other_algorithm_list = None,
                                           KNN_similarity_list = KNN_similarity_to_report_list,
                                           ICM_names_list = dataSplitter.get_loaded_ICM_dict().keys(),
                                           UCM_names_list = dataSplitter.get_loaded_UCM_dict().keys(),
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


