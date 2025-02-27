"""
Created on 27/09/2019

@author: Maurizio Ferrari Dacrema
"""


from Recommenders.Recommender_import_list import *
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender


class ConfigTuple(object):

    def __init__(self,
                 recommender_name = None,
                 recommender_class = None,
                 KNN_similarity = None,
                 ICM_name = None,
                 UCM_name = None,
                 ):

        assert recommender_name is not None
        assert recommender_class is not None

        self.recommender_name = recommender_name
        self.recommender_class = recommender_class
        self.KNN_similarity = KNN_similarity
        self.ICM_name = ICM_name
        self.UCM_name = UCM_name

    def __eq__(self, other):
        result = self.recommender_class == other.recommender_class and \
                self.KNN_similarity == other.KNN_similarity and \
                self.ICM_name == other.ICM_name and \
                self.UCM_name == other.UCM_name

        return result


class RecommenderConfigurationTupleIterator(object):
    """RecommenderConfigurationTupleIterator"""

    def __init__(self,
                 recommender_class_list = None,
                 KNN_similarity_list = None,
                 ICM_name_list = None,
                 UCM_name_list = None,
                 ):

        self._recommender_class_list = recommender_class_list
        self._recommender_name_list = []
        self._name_to_configuration_tuple_dict = {}
        self._KNN_similarity_list = KNN_similarity_list
        self._ICM_name_list = ICM_name_list
        self._UCM_name_list = UCM_name_list

        self._build_configuration_tuple_list()

    def get_configuration_from_name(self, recommender_name):
        return self._name_to_configuration_tuple_dict[recommender_name]

    def _build_configuration_tuple_list(self):

        self._recommender_name_list = []
        self._name_to_configuration_tuple_dict = {}

        for recommender_class in self._recommender_class_list:

            if issubclass(recommender_class, BaseItemCBFRecommender):

                for ICM_name in self._ICM_name_list:
                    ICM_label = "_{}".format(ICM_name)

                    if recommender_class in [ItemKNNCBFRecommender, ItemKNN_CFCBF_Hybrid_Recommender]:
                        for KNN_similarity in self._KNN_similarity_list:
                            recommender_name = recommender_class.RECOMMENDER_NAME + ICM_label + "_{}".format(KNN_similarity)

                            self._recommender_name_list.append(recommender_name)
                            self._name_to_configuration_tuple_dict[recommender_name] = ConfigTuple(recommender_name, recommender_class, KNN_similarity, ICM_name, None)
                    else:
                        recommender_name = recommender_class.RECOMMENDER_NAME + ICM_label

                        self._recommender_name_list.append(recommender_name)
                        self._name_to_configuration_tuple_dict[recommender_name] = ConfigTuple(recommender_name, recommender_class, None, ICM_name, None)


            elif issubclass(recommender_class, BaseUserCBFRecommender):

                for UCM_name in self._UCM_name_list:
                    UCM_label = "_{}".format(UCM_name)

                    if recommender_class in [UserKNNCBFRecommender, UserKNN_CFCBF_Hybrid_Recommender]:
                        for KNN_similarity in self._KNN_similarity_list:
                            recommender_name = recommender_class.RECOMMENDER_NAME + UCM_label + "_{}".format(KNN_similarity)

                            self._recommender_name_list.append(recommender_name)
                            self._name_to_configuration_tuple_dict[recommender_name] = ConfigTuple(recommender_name, recommender_class, KNN_similarity, None, UCM_name)
                    else:
                        recommender_name = recommender_class.RECOMMENDER_NAME + UCM_label

                        self._recommender_name_list.append(recommender_name)
                        self._name_to_configuration_tuple_dict[recommender_name] = ConfigTuple(recommender_name, recommender_class, None, None, UCM_name)


            else:

                if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:

                    for KNN_similarity in self._KNN_similarity_list:
                        recommender_name = recommender_class.RECOMMENDER_NAME + "_{}".format(KNN_similarity)

                        self._recommender_name_list.append(recommender_name)
                        self._name_to_configuration_tuple_dict[recommender_name] = ConfigTuple(recommender_name, recommender_class, KNN_similarity, None, None)

                else:
                    recommender_name = recommender_class.RECOMMENDER_NAME
                    self._recommender_name_list.append(recommender_name)
                    self._name_to_configuration_tuple_dict[recommender_name] = ConfigTuple(recommender_name, recommender_class, None, None, None)



    def __iter__(self):
        self._instance_index = 0
        return self

    def __next__(self):
        if self._instance_index < len(self._recommender_name_list):
            recommender_name = self._recommender_name_list[self._instance_index]
            self._instance_index += 1
            return self.get_configuration_from_name(recommender_name)
        else:
            raise StopIteration

    def len(self):
        return len(self._name_to_configuration_tuple_dict)

    def rewind(self):
        self._instance_index = 0




class RecommenderInstanceIterator(object):
    """RecommenderInstanceIterator"""

    def __init__(self,
                 recommender_class_list = None,
                 KNN_similarity_list = None,
                 URM = None,
                 ICM_dict = None,
                 UCM_dict = None,
                 ):

        assert URM is not None, "RecommenderInstanceIterator: Model requires URM, which is None."

        self._recommender_class_list = recommender_class_list
        self._recommender_instance_list = []
        self._recommender_name_list = []
        self._name_to_instance_dict = {}
        self._KNN_similarity_list = KNN_similarity_list
        self._URM = URM
        self._ICM_dict = ICM_dict
        self._UCM_dict = UCM_dict
        self._configuration_iterator = None


    def get_instance_from_name(self, recommender_name):
        return self._name_to_instance_dict[recommender_name]

    def get_configuration_from_name(self, recommender_name):
        return self._configuration_iterator._name_to_configuration_tuple_dict[recommender_name]

    def __iter__(self):
        self._configuration_iterator = RecommenderConfigurationTupleIterator(recommender_class_list = self._recommender_class_list,
                                                                              KNN_similarity_list = self._KNN_similarity_list,
                                                                              ICM_name_list = self._ICM_dict.keys(),
                                                                              UCM_name_list = self._UCM_dict.keys(),
                                                                              )

        for model_tuple in self._configuration_iterator:

            if model_tuple.ICM_name is not None:
                ICM_object = self._ICM_dict[model_tuple.ICM_name]
                recommender_instance = model_tuple.recommender_class(self._URM, ICM_object)
            elif model_tuple.UCM_name is not None:
                UCM_object = self._UCM_dict[model_tuple.UCM_name]
                recommender_instance = model_tuple.recommender_class(self._URM, UCM_object)
            else:
                recommender_instance = model_tuple.recommender_class(self._URM)

            yield recommender_instance, model_tuple.recommender_name


    def len(self):
        return self._configuration_iterator.len()

    def rewind(self):
        if self._configuration_iterator is not None:
            self._configuration_iterator.rewind()







class RecommenderInstanceIterator_Light(object):
    """RecommenderInstanceIterator"""

    def __init__(self,
                 recommender_class_list = None,
                 KNN_similarity_list = None,
                 URM = None,
                 ICM_dict = None,
                 UCM_dict = None,
                 ):

        assert URM is not None, "RecommenderInstanceIterator: Model requires URM, which is None."

        self._recommender_class_list = recommender_class_list
        self._KNN_similarity_list = KNN_similarity_list
        self._URM = URM
        self._ICM_dict = ICM_dict
        self._UCM_dict = UCM_dict


    def __iter__(self):

        for recommender_class in self._recommender_class_list:

            if issubclass(recommender_class, BaseItemCBFRecommender):

                for ICM_name, ICM_object in self._ICM_dict.items():
                    ICM_label = "_{}".format(ICM_name)

                    for KNN_similarity in self._KNN_similarity_list:
                        recommender_instance = recommender_class(self._URM, ICM_object)
                        recommender_name = recommender_instance.RECOMMENDER_NAME + ICM_label + "_" + KNN_similarity

                        yield recommender_instance, recommender_name

            elif issubclass(recommender_class, BaseUserCBFRecommender):

                for UCM_name, UCM_object in self._UCM_dict.items():
                    UCM_label = "_{}".format(UCM_name)

                    for KNN_similarity in self._KNN_similarity_list:
                        recommender_instance = recommender_class(self._URM, UCM_object)
                        recommender_name = recommender_instance.RECOMMENDER_NAME + UCM_label + "_" + KNN_similarity

                        yield recommender_instance, recommender_name

            else:

                if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:

                    for KNN_similarity in self._KNN_similarity_list:
                        recommender_instance = recommender_class(self._URM)
                        recommender_name = recommender_instance.RECOMMENDER_NAME + "_" + KNN_similarity

                        yield recommender_instance, recommender_name

                else:
                    recommender_instance = recommender_class(self._URM)
                    recommender_name = recommender_instance.RECOMMENDER_NAME

                    yield recommender_instance, recommender_name


