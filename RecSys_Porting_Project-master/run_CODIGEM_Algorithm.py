#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os, traceback, argparse, multiprocessing
import numpy as np
from functools import partial

from Conferences.RecSysProject.CODIGEM_our_interface.ML20Reader import ML20Reader
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from Recommenders.Recommender_import_list import *
from Utils.ResultFolderLoader import ResultFolderLoader
from Evaluation.Evaluator import EvaluatorHoldout
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from HyperparameterTuning.hyperparameter_space_library import ExperimentConfiguration
from Utils.RecommenderInstanceIterator import RecommenderConfigurationTupleIterator
from HyperparameterTuning.functions_for_parallel_model import _unpack_tuple_and_search
from Conferences.RecSysProject.CODIGEM_our_interface.ML20Wrapper import ML20Wrapper


def read_data_split_and_search(dataset_name, flag_baselines_tune=False, flag_model_default=False,
                               flag_print_results=False):
    """
    Funzione principale per il caricamento dei dati, la ricerca degli iperparametri e l'addestramento del modello.
    """

    result_folder_path = f"result_experiments/{CONFERENCE_NAME}/{ALGORITHM_NAME}_{dataset_name}/"
    model_folder_path = os.path.join(result_folder_path, "data_split/")

    # Caricamento del dataset Movielens20M
    dataset = ML20Reader(result_folder_path)
    print(f'Current dataset: {dataset_name}')

    # Estrarre URM train, validation e test
    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_train_last_test = URM_train + URM_validation

    # Assicurarsi che i dati siano impliciti e le matrici di train/test siano disgiunte
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # Creare la cartella per i risultati se non esiste
    os.makedirs(result_folder_path, exist_ok=True)

    # Parametri per la valutazione del modello
    metric_to_optimize = 'NDCG'
    cutoff_to_optimize = 10
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]

    # Definizione dell'evaluator per la validazione e il test
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    # Addestramento del modello avanzato
    if flag_model_default:
        try:
            article_hyperparameters = {
                "batch_size": 512,
                "epochs": 300,
                "embedding_size": 64,
                "hidden_size": 128,
                "negative_sample_per_positive": 1,
                "regularization_users_items": 0.01,
                "regularization_weights": 10,
                "learning_rate_embeddings": 0.05,
                "learning_rate_CNN": 0.05,
                "channel_size": [32] * 6,
                "dropout": 0.0,
                "epoch_verbose": 1,
            }

            earlystopping_hyperparameters = {
                "validation_every_n": 5,
                "stop_on_validation": True,
                "lower_validations_allowed": 5,
                "evaluator_object": evaluator_validation,
                "validation_metric": metric_to_optimize,
            }

            hyperparameterSearch = SearchSingleCase(ML20Wrapper, evaluator_validation, evaluator_test)
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                FIT_KEYWORD_ARGS={},
                EARLYSTOPPING_KEYWORD_ARGS=earlystopping_hyperparameters,
            )
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test

            hyperparameterSearch.search(
                recommender_input_args,
                recommender_input_args_last_test,
                fit_hyperparameters_values={},
                metric_to_optimize=metric_to_optimize,
                cutoff_to_optimize=cutoff_to_optimize,
                output_folder_path=model_folder_path,
                output_file_name_root=ML20Wrapper.RECOMMENDER_NAME,
                resume_from_saved=True,
                save_model="best",
                evaluate_on_test="best",
            )

        except Exception as e:
            print(f"Exception in {ML20Wrapper}: {str(e)}")
            traceback.print_exc()

    # Addestramento dei modelli baseline
    if flag_baselines_tune:
        recommender_class_list = [Random, TopPop, GlobalEffects, ItemKNNCFRecommender, PureSVDRecommender]
        experiment_configuration = ExperimentConfiguration(
            URM_train=URM_train,
            URM_train_last_test=URM_train_last_test,
            ICM_DICT=dataset.ICM_DICT,
            UCM_DICT=dataset.UCM_DICT,
            evaluator_validation=evaluator_validation,
            evaluator_test=evaluator_test,
            evaluator_validation_earlystopping=evaluator_validation_earlystopping,
            metric_to_optimize=metric_to_optimize,
            cutoff_to_optimize=cutoff_to_optimize,
            n_cases=50,
            n_random_starts=16,
            n_processes=3,
            save_model="last",
            evaluate_on_test="best",
        )

        configuration_iterator = RecommenderConfigurationTupleIterator(recommender_class_list)
        _unpack_tuple_and_search_partial = partial(_unpack_tuple_and_search, experiment_configuration,
                                                   model_folder_path)

        with multiprocessing.Pool(processes=experiment_configuration.n_processes) as pool:
            pool.map(_unpack_tuple_and_search_partial, configuration_iterator, chunksize=1)

    # Stampa dei risultati
    if flag_print_results:
        result_loader = ResultFolderLoader(model_folder_path)
        result_loader.generate_latex_results(os.path.join(result_folder_path, "accuracy_metrics.txt"),
                                             ['RECALL', 'PRECISION', 'MAP', 'NDCG'], [cutoff_to_optimize])
        result_loader.generate_latex_results(os.path.join(result_folder_path, "beyond_accuracy_metrics.txt"),
                                             ["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM"], cutoff_list)


if __name__ == '__main__':
    ALGORITHM_NAME = "CODIGEM"
    CONFERENCE_NAME = "Movielens20"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune', help="Hyperparameter tuning for baselines", type=bool, default=False)
    parser.add_argument('-m', '--model_default', help="Train default model", type=bool, default=False)
    parser.add_argument('-p', '--print_results', help="Print results", type=bool, default=True)

    input_flags = parser.parse_args()
    dataset_list = ["movielens20m"]
    for dataset_name in dataset_list:
        read_data_split_and_search(dataset_name, input_flags.baseline_tune, input_flags.model_default,
                                   input_flags.print_results)
