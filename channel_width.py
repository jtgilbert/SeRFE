""" Uses channel geometry relationships with a brief table of training data to attribute each segment of
an input stream network with values for channel width. """


# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import geopandas as gpd
import os


def get_width_params(width_table):
    """
    Uses regression to obtain a and b values in the hydraulic geometry for channel width: w=aQ^b
    :param width_table: csv - column 1 header: 'w_low', column 2 header: 'w_bf', column 3 header: 'w_flood',
    column 4 header: 'DA (km2)'
    :return: float - values of a and b
    """
    table = pd.read_csv(width_table, sep=',', header=0)
    table = table.dropna(axis='columns')
    logw_low = np.asarray(np.log10(table['w_low'])).reshape(-1, 1)
    logw_bf = np.asarray(np.log10(table['w_bf'])).reshape(-1, 1)
    logw_flood = np.asarray(np.log10(table['w_flood'])).reshape(-1, 1)
    logDA = np.asarray(np.log10(table['DA (km2)'])).reshape(-1, 1)

    # low flow width regression
    regr = linear_model.LinearRegression()
    regr.fit(logDA, logw_low)
    rsq = regr.score(logDA, logw_low)
    a_low = float(10**regr.intercept_)
    b_low = float(regr.coef_)

    # bankfull width regression
    regr2 = linear_model.LinearRegression()
    regr2.fit(logDA, logw_bf)
    rsq2 = regr2.score(logDA, logw_bf)
    a_bf = float(10**regr2.intercept_)
    b_bf = float(regr2.coef_)

    # flood width regression
    regr3 = linear_model.LinearRegression()
    regr3.fit(logDA, logw_flood)
    rsq3 = regr3.score(logDA, logw_flood)
    a_flood = float(10**regr3.intercept_)
    b_flood = float(regr3.coef_)

    # produce plots (consider putting all info onto one plot instead of 3)
    x = np.linspace(0,np.max(10**logDA))
    y_low = (a_low*x**b_low).reshape(-1, 1)
    y_bf = (a_bf*x**b_bf).reshape(-1, 1)
    y_flood = (a_flood*x**b_flood).reshape(-1, 1)

    plotname = width_table[0:-4]+"_plot.png"
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 14))
    ax1.scatter(10**logDA, 10**logw_low, color='k')
    ax1.plot(x, y_low, color='blue')
    ax1.set_title("Low Flow Channel Width", fontsize='medium', fontweight='bold')
    ax1.set_xlabel("Drainage Area (km2)")
    ax1.set_ylabel("Channel Width (m)")
    ax1.grid(color="lightgrey")
    ax1.text(np.max(10**logDA-(10**logDA/2)), 1, 'R-squared: {0:.2f} \nEquation: w={1:.2f}DA^{2:.2f}'
             .format(rsq, a_low, b_low))
    ax2.scatter(10 ** logDA, 10 ** logw_bf, color='k')
    ax2.plot(x, y_bf, color='blue')
    ax2.set_title("Bankfull Channel Width", fontsize='medium', fontweight='bold')
    ax2.set_xlabel("Drainage Area (km2)")
    ax2.set_ylabel("Channel Width (m)")
    ax2.grid(color="lightgrey")
    ax2.text(np.max(10 ** logDA - (10 ** logDA / 2)), 1, 'R-squared: {0:.2f} \nEquation: w={1:.2f}DA^{2:.2f}'
             .format(rsq2, a_bf, b_bf))
    ax3.scatter(10 ** logDA, 10 ** logw_flood, color='k')
    ax3.plot(x, y_flood, color='blue')
    ax3.set_title("Flooded Channel Width", fontsize='medium', fontweight='bold')
    ax3.set_xlabel("Drainage Area (km2)")
    ax3.set_ylabel("Channel Width (m)")
    ax3.grid(color="lightgrey")
    ax3.text(np.max(10 ** logDA - (10 ** logDA / 2)), 1, 'R-squared: {0:.2f} \nEquation: w={1:.2f}DA^{2:.2f}'
             .format(rsq3, a_flood, b_flood))

    fig.savefig(plotname, dpi=150)

    return a_low, b_low, a_bf, b_bf, a_flood, b_flood


def add_w(network, a_low, b_low, a_bf, b_bf, a_flood, b_flood, crs_epsg):
    """
    Applies the values of a and b obtained from 'get_width_params' function to hydraulic geometry equation
    for width to attribute channel width to drainage network shapefile
    :param network: string - path to input drainage network (must already be attributed with drainage area values)
    :param a: float - obtained from function 'get_width_params'
    :param b: float - obtained from function 'get_width_params'
    :param crs_epsg: int - epsg number for coordinate reference system
    :return:
    """
    # convert epsg number into crs dict
    sref = {'init': 'epsg:{}'.format(crs_epsg)}

    # read in network and check for projection
    flowlines = gpd.read_file(network)
    if flowlines['geometry'].crs == sref:
        pass
    else:
        flowlines = flowlines.to_crs(sref)

    # populate a new width field based on the parameters derived from table
    flowlines['w_low'] = a_low*flowlines['Drain_Area']**b_low

    flowlines['w_bf'] = a_bf*flowlines['Drain_Area']**b_bf

    flowlines['w_flood'] = a_flood*flowlines['Drain_Area']**b_flood

    # correct any headwater anomalies
    for i in flowlines.index:
        if flowlines.loc[i, 'w_bf'] < flowlines.loc[i, 'w_low']:
            flowlines.loc[i, 'w_bf'] = flowlines.loc[i, 'w_low']
        if flowlines.loc[i, 'w_flood'] < flowlines.loc[i, 'w_bf']:
            flowlines.loc[i, 'w_flood'] = flowlines.loc[i, 'w_bf']

    flowlines.to_file(network)

    return
