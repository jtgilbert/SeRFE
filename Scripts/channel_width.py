""" Uses channel geometry relationships with a brief table of training data to attribute each segment of
an input stream network with values for channel width. """


# imports
import numpy as np
from sklearn import linear_model
import pandas as pd
import geopandas as gpd


def get_width_model(width_table):
    """
    Uses regression to obtain a model for predicting width based on drainage area and discharge
    :param width_table: csv - column 1 header: 'DA', column 2 header: 'Q', column 3 header: 'w'
    :return: regression model object
    """
    table = pd.read_csv(width_table, sep=',', header=0)
    table = table.dropna(axis='columns')
    table['DA'] = np.log(table['DA'])
    table['Q'] = np.sqrt(table['Q'])

    # width regression
    regr = linear_model.LinearRegression()
    regr.fit(table[['DA', 'Q']], table['w'])
    rsq = regr.score(table[['DA', 'Q']], table['w'])
    if rsq < 0.5:
        print 'R-squared is less than 0.5, poor model fit'

    return regr
