""" This class provides the functions to parameterize flow estimates and attribute them to drainage networks. """

import numpy as np
from sklearn import linear_model
import pandas as pd
import geopandas as gpd
import network_topology as nt
import matplotlib.pyplot as plt
import os


class HistoricQ:
    """
    Uses a table of drainage area and corresponding Q2 values from 'least-disturbed' gauges in a basin to find
    parameter values for a regression to predict historic Q2 in ungauged locations (currently set up as linear
    regression i.e. Q2 is linear function of drainage area, based on Santa Clara River watershed). The drainage
    network is then attributed with a historic value of Q2 for each segment.
    """

    def __init__(self, network, Q2_table):
        """
        :param network: stream network shapefile
        :param Q2_table: csv with fields 'Q2 (cms)' and 'DA (km2)' and optionally 'MAP (cm)'
        """
        self.streams = network
        self.network = gpd.read_file(network)
        self.Q2_table = Q2_table
        table = pd.read_csv(Q2_table, sep=',', header=0)
        table = table.dropna(axis=0)
        self.gauges = table['Gauge']
        self.Q2 = table['Q2 (cms)']
        self.DA = table['DA (km2)']
        if table.shape[1] == 4:
            self.precip = table['MAP (cm)']

        a, b = self.q2_equn_pow_1param()
        self.find_q2(a, b)
        #self.q2_to_network(q2_vals)

    def q2_equn_pow_1param(self):
        """
        obtains the parameters for the linear equation Q2 = A*Drainage Area + B.
        :return:
        """

        # log transform data to linearize
        logQ2 = np.array(np.log10(self.Q2), dtype=np.float16).reshape(-1, 1)
        logDA = np.array(np.log10(self.DA), dtype=np.float16).reshape(-1, 1)

        # fit linear model
        reg = linear_model.LinearRegression(fit_intercept=True)
        reg.fit(logDA, logQ2)
        pred = reg.predict(logDA)
        r_squ = reg.score(logDA, logQ2)

        # retrieve parameters to return
        a = 10**reg.intercept_
        b = reg.coef_

        # produce plot
        plotname = os.path.basename(self.Q2_table)[0:-4] + "_plot.png"
        plotpath = os.path.dirname(self.Q2_table)
        x = np.linspace(0, np.max(self.DA))
        y = (a * x ** b).reshape(-1, 1)
        plt.xlim((0, np.max(self.DA) + 10))
        plt.ylim((0, np.max(self.Q2) + 10))
        plt.scatter(self.DA, self.Q2, color='k')
        plt.plot(x, y, color='blue')
        plt.xlabel('Drainage Area (km2)')
        plt.ylabel('Q2 (cms)')
        plt.grid(color='lightgrey')
        plt.text(np.max(10**logDA-(10**logDA/2)), np.max(10**logQ2-50),
                 'R-squared: {0:.2f} \nEquation: Q2={1:.2f}DA^{2:.2f}'.format(float(r_squ), float(a), float(b)))
        plt.savefig(plotpath + "/" + plotname, dpi=150)

        return a, b

    def find_q2(self, a, b):
        """
        Attributes each segment of stream network with historic Q2 value.
        Applies equation q2 = a*da**b
        :param a: 'a' parameter
        :param b: 'b' parameter
        :return: q2 value
        """

        for i in self.network.index:
            da = self.network.loc[i, 'Drain_Area']
            q2 = float(a) * da ** float(b)
            self.network.loc[i, 'Q2 (cms)'] = q2

        self.network.to_file(self.streams)

        return


class UpdatedQ:
    """
    Takes an input table with stream segment IDs and their current (altered) Q2 values from disturbance (eg a dam)
    and re-attributes the whole drainage network with updated Q2 values.
    """

    def __init__(self, network, updated_Q2_table, a=None, b=None):
        """
        :param network: stream network shapefile
        :param updated_Q2_table: csv file with stream segment id and new, post-disturbance Q2 value
        """

        self.streams = network
        self.a = a
        self.b = b
        self.network = gpd.read_file(self.streams)
        table = pd.read_csv(updated_Q2_table, sep=',', header=0)
        table = table.dropna(axis=0)
        self.segid = table['Segment ID']
        self.newQ2 = table['Updated Q2 (cms)']

        self.topo = nt.TopologyTools(network)

    def q2_additions(self, seg, dist_da):
        """

        :param seg:
        :param dist_da:
        :return:
        """

        dr_area = self.network.loc[seg, 'Drain_Area'] - dist_da
        q2_add = self.a * dr_area**self.b

        if q2_add >= 0:
            return q2_add
        else:
            return 0

    def update_affected_reaches(self):
        """
        Attributes segments with new Q2 values from alteration downstream to next confluence.
        :return:
        """

        for x in range(len(self.segid)):
            self.network.loc[self.segid[x], 'newQ2'] = self.newQ2[x]
            next_reach = self.topo.get_next_reach(self.segid[x])
            while next_reach is not None:
                if self.network.loc[next_reach, 'confluence'] == 0:
                    dist_da = self.network.loc[self.segid[x], 'Drain_Area']
                    q2_add = self.q2_additions(next_reach, dist_da)
                    self.network.loc[next_reach, 'newQ2'] = self.newQ2[x] + q2_add
                    next_reach = self.topo.get_next_reach(next_reach)
                else:
                    next_reach = None

        self.network.to_file(self.streams)

        return

    def update_above_dist(self):
        """
        Attributes segments wtih new Q2 values that are the same as the old Q2 values upstream of any disturbance.
        :return:
        """

        dist_us = []
        dist_ds = []

        for x in range(len(self.segid)):
            dist_x_us = self.topo.find_all_us(self.segid[x])
            for seg in dist_x_us:
                dist_us.append(seg)
            dist_x_ds = self.topo.find_all_ds(self.segid[x])
            for seg in dist_x_ds:
                dist_ds.append(seg)

        final_us = []
        for seg in dist_us:
            if seg not in dist_ds:
                final_us.append(seg)

        for i in self.network.index:
            if i in final_us:
                self.network.loc[i, 'newQ2'] = self.network.loc[i, 'Q2 (cms)']

        self.network.to_file(self.streams)

        return

    def update_below_confluences(self):
        """
        Attributes segments downstream of confluences in network with new Q2 values.
        :return:
        """

        conf_list = []

        for i in self.network.index:
            if self.network.loc[i, 'confluence'] == 1:
                if self.network.loc[i, 'newQ2'] == -9999:
                    conf_list.append(i)

        while len(conf_list) > 0:
            for x in conf_list:

                us1 = self.topo.find_us_seg(x)
                us1q2 = self.network.loc[us1, 'newQ2']
                us2 = self.topo.find_us_seg2(x)
                us2q2 = self.network.loc[us2, 'newQ2']
                if us1q2 != -9999 and us2q2 != -9999:

                    self.network.loc[x, 'newQ2'] = us1q2 + us2q2

                    next_reach = self.topo.get_next_reach(x)
                    while next_reach is not None:

                        if self.network.loc[next_reach, 'confluence'] == 0:
                            if self.network.loc[next_reach, 'newQ2'] == -9999:
                                us_reach = self.topo.find_us_seg(next_reach)
                                us_q2 = self.network.loc[us_reach, 'newQ2']
                                us_da = self.network.loc[us_reach, 'Drain_Area']
                                q2_addon = self.q2_additions(next_reach, us_da)
                                self.network.loc[next_reach, 'newQ2'] = us_q2 + q2_addon
                                next_reach = self.topo.get_next_reach(next_reach)
                            else:
                                next_reach = None
                        else:
                            next_reach = None

                    conf_list.remove(x)

                else:
                    pass

        self.network.to_file(self.streams)

        return

wd = '/home/jordan/Documents/Geoscience/jupyter_testing/Piru_test/'
network = wd + 'Piru_network_500m.shp'
q2table = wd + 'referenceQ2.csv'

HistoricQ(network, q2table)
