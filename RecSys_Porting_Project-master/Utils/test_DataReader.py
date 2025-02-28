#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/03/18

@author: Maurizio Ferrari Dacrema
"""
import sys
sys.path.append('c:/Users/laral/Desktop/Progetto RS/RS/RecSys_Porting_Project-master')

# Check whether they work correctly

from Data_manager.TheMoviesDataset.TheMoviesDatasetReader import TheMoviesDatasetReader
#from Data_manager.NetflixEnhanced.NetflixEnhancedReader import NetflixEnhancedReader
from Data_manager.BookCrossing.BookCrossingReader import BookCrossingReader
from Data_manager.XingChallenge2016.XingChallenge2016Reader import XingChallenge2016Reader
from Data_manager.XingChallenge2017.XingChallenge2017Reader import XingChallenge2017Reader
from Data_manager.AmazonReviewData.AmazonBooksReader import AmazonBooksReader
from Data_manager.AmazonReviewData.AmazonAutomotiveReader import AmazonAutomotiveReader
from Data_manager.AmazonReviewData.AmazonElectronicsReader import AmazonElectronicsReader
#from Data_manager.AmazonReviewData.AmazonInstantVideo import AmazonInstantVideoReader
from Data_manager.AmazonReviewData.AmazonMusicalInstrumentsReader import AmazonMusicalInstrumentsReader
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.Movielens.Movielens20MReader import Movielens20MReader
from Data_manager.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader
from Data_manager.ThirtyMusic.ThirtyMusicReader import ThirtyMusicReader


from Data_manager.DataSplitter import DataSplitter
import traceback


def read_split_for_data_reader(dataReader_class, force_new_split = False):

    dataReader = dataReader_class()
    dataSplitter = DataSplitter(dataReader_class, ICM_to_load=None, force_new_split=force_new_split)

    URM_train = dataSplitter.get_URM_train()
    URM_validation = dataSplitter.get_URM_validation()
    URM_test = dataSplitter.get_URM_test()




dataReader_list = [
    Movielens1MReader,
    Movielens20MReader,
    NetflixPrizeReader,
    TheMoviesDatasetReader,
    BookCrossingReader,
    #NetflixEnhancedReader,
    XingChallenge2016Reader,
    XingChallenge2017Reader,
    AmazonAutomotiveReader,
    AmazonBooksReader,
    AmazonMusicalInstrumentsReader,
    #AmazonInstantVideoReader,
    AmazonElectronicsReader,
    ThirtyMusicReader,

]

test_list = []


for dataReader_class in dataReader_list:

    try:

        read_split_for_data_reader(dataReader_class, force_new_split = False)

        print("Test for: {} - OK".format(dataReader_class))
        test_list.append((dataReader_class, "OK"))


    except Exception as exception:

        traceback.print_exc()

        print("Test for: {} - Trying to generate new split".format(dataReader_class))

        try:

            read_split_for_data_reader(dataReader_class, force_new_split = True)

            print("Test for: {} - OK".format(dataReader_class))
            test_list.append((dataReader_class, "OK"))


        except Exception as exception:

            traceback.print_exc()

            print("Test for: {} - FAIL".format(dataReader_class))
            test_list.append((dataReader_class, "FAIL"))






print("\n\n\n\nSUMMARY:")

for dataReader_class, outcome in test_list:

    print("Test for: {} - {}".format(dataReader_class, outcome))