"""
Notes
-----
This is a utility file consists of custom classes.
"""
import configparser

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

config = configparser.ConfigParser()
config.read("setup.cfg")


class AdditionalAttributes(BaseEstimator, TransformerMixin):
    """Class to transform dataframe by adding new columns to dataframe.

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_ix = int(config["Data"]["rooms_ix"])
        bedrooms_ix = int(config["Data"]["bedrooms_ix"])
        population_ix = int(config["Data"]["population_ix"])
        households_ix = int(config["Data"]["households_ix"])
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room
        ]
