import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import os
import network_topology as nt
from sklearn import linear_model



class Dpred:
    """
    This class uses field measurements of grain size distributions to estimate a critical discharge value for each
    segment of a drainage network. This value is then used to predict grain size throughout the network.
    """

    def __init__(self, network, grain_size, reach_ids, width_table):
        """

        :param network: drainage network shapefile
        :param grain_size: list of paths to csv's of grain size measurements in less-disturbed areas
        :param f_sand: list of the fraction of sand in the bed at each site where grain size was measured; the order
        should be teh same as the order of the list of grain size measurement csv files.
        :param reach_ids: the drainage network reach ids associated with the locations of grain size sampling; the
        order should be the same as the order of the list of grain size measurement csv files.
        """

        self.streams = network
        self.network = gpd.read_file(network)
        self.grain_size = grain_size
        self.reach_ids = reach_ids
        self.width_table = width_table
        self.topo = nt.TopologyTools(self.streams)

        self.width = []
        self.slope = []
        self.Q2 = []
        self.Qc_mid = []
        self.Qc_low = []
        self.Qc_high = []

        # add a 'bankfull' width attribute
        w_model = self.get_width_model()
        self.add_w_bf(w_model)

        self.network = gpd.read_file(self.streams)

        for i in self.reach_ids:
            w = self.network.loc[i, 'w_bf']
            self.width.append(w)
            s = self.network.loc[i, 'Slope_mid']
            self.slope.append(s)
            q2 = self.network.loc[i, 'Q2 (cms)']
            self.Q2.append(q2)

        for x in range(len(self.reach_ids)):
            qc_mid, qc_low, qc_high = self.find_Qc(x)
            self.Qc_mid.append(qc_mid)
            self.Qc_low.append(qc_low)
            self.Qc_high.append(qc_high)

        self.Qc_prop_mid = []
        self.Qc_prop_low = []
        self.Qc_prop_high = []
        for y in range(len(self.Q2)):
            prop_mid = self.Qc_mid[y] / self.Q2[y]
            self.Qc_prop_mid.append(prop_mid)
            prop_low = self.Qc_low[y] / self.Q2[y]
            self.Qc_prop_low.append(prop_low)
            prop_high = self.Qc_high[y] / self.Q2[y]
            self.Qc_prop_high.append(prop_high)

        self.mean_prop_mid = np.mean(self.Qc_prop_mid)
        self.mean_prop_low = np.mean(self.Qc_prop_low)
        self.mean_prop_high = np.mean(self.Qc_prop_high)

        self.add_Qc()
        self.find_Dpred()

    def get_width_model(self):
        """
        Uses regression to obtain a model for predicting width based on drainage area and discharge
        :param width_table: csv - column 1 header: 'DA', column 2 header: 'Q', column 3 header: 'w'
        :return: regression model object
        """
        table = pd.read_csv(self.width_table, sep=',', header=0)
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

    def add_w_bf(self, model):
        w_bf = []
        for i in self.network.index:
            w_inputs = np.array([np.log(self.network.loc[i, 'Drain_Area']), self.network.loc[i, 'Q2 (cms)']**0.5])
            w_pred = model.predict([w_inputs])[0]
            w_bf.append(w_pred)

        self.network['w_bf'] = w_bf
        self.network.to_file(self.streams)

        return

    def find_Qc(self, item):
        """

        """

        # set up constants
        om_crit_star = 0.1
        g = 9.81
        rho = 1000
        rho_s = 2650

        # read in grain size array
        D_array = pd.read_csv(self.grain_size[item], delimiter=',', header=0)

        # produce percent finer grain size plot
        plotname = os.path.basename(self.grain_size[item])[0:-4]+"_plot.png"
        plotpath = os.path.dirname(self.grain_size[item])
        dsort = D_array.sort_values(by=['D'])
        rank = np.arange(1, len(dsort) + 1, 1)
        dsort['rank'] = rank
        dsort['%finer'] = (dsort['rank'] / len(dsort)) * 100

        fig, ax = plt.subplots()
        ax.plot(dsort['D'], dsort['%finer'], color='red', label='_nolegend_')
        ax.axvline(dsort.median()[0], linestyle='dashed', color='k', label='Median = '+'{0:.2f}'.format(dsort.median()[0]))
        plt.xlabel('Grain Size (mm)')
        plt.ylabel('% Finer')
        plt.grid(color='lightgrey')
        plt.legend(loc=4)
        fig.suptitle('Grain size')
        fig.savefig(plotpath + "/" + plotname, dpi=150)

        # subset grain size array to remove high outliers
        gs = []
        for x in D_array['D']:
            if x <= D_array['D'].quantile(0.95):  # removes high outliers
                gs.append(x)
        gs = np.asarray(gs)

        # bootstrap data to produce range of median grain size values, return mean of these values
        num_boots = 100
        boot_medians = []
        for i in range(num_boots):
            boot = resample(gs, replace=True, n_samples=len(gs))
            median = np.median(boot)
            boot_medians.append(median)
        medgr = np.asarray(boot_medians)

        medgs = np.zeros_like(medgr)
        for i in range(len(medgr)):
            medgs[i] = medgr[i] / 1000.

        Qc_values = []

        for i in range(len(medgs)):
            numerator = om_crit_star * self.width[item] * (rho_s - rho) * ((((rho_s / rho) - 1.) * g * (medgs[i] ** 3.)) ** 0.5)
            denominator = rho * self.slope[item]
            Qc = numerator / denominator

            Qc_values.append(Qc)
        Qc_values = np.asarray(Qc_values)

        meanQc = np.mean(Qc_values)
        lowQc = meanQc - (1.64 * np.std(Qc_values))
        highQc = meanQc + (1.64 * np.std(Qc_values))

        # create plot of median grain size values and Qc values
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
        ax1.hist(medgr, bins=15, color='blue', edgecolor='k', label='_nolegend_')
        ax1.axvline(np.mean(medgr), color='k', linestyle='dashed', label='Mean = ' + '{0:.2f}'.format(np.mean(medgr)))
        ax1.set_title("Bootstrap Medians", fontsize='x-large', fontweight='bold')
        ax1.set_xlabel("Median Grain Size (mm)")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax2.hist(Qc_values, bins=15, color='seagreen', edgecolor='k', label='_nolegend_')
        ax2.axvline(meanQc, color='k', linestyle='dashed', label='Mean = ' + '{0:.2f}'.format(np.mean(Qc_values)))
        ax2.axvline(lowQc, color='red', linestyle='dashed', label='90% prediction interval')
        ax2.axvline(highQc, color='red', linestyle='dashed', label='_nolegend_')
        ax2.set_title("Critical Discharge Values", fontsize='x-large', fontweight='bold')
        ax2.set_xlabel("Critical Discharge (m3/s)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        fig.savefig(plotpath + "/" + os.path.basename(self.grain_size[item])[0:-4] + "_Qc_plot.png")

        return meanQc, lowQc, highQc

    def add_Qc(self):  # can vectorize this
        """
        Adds a 'Qc' critical discharge field to drainage network attribute table informed by measured grain size
        distributions
        :return:
        """

        network = gpd.read_file(self.streams)

        network['Qc_mid'] = network['Q2 (cms)']*self.mean_prop_mid  # is this best way to extrapolate??
        network['Qc_low'] = network['Q2 (cms)']*self.mean_prop_low
        network['Qc_high'] = network['Q2 (cms)']*self.mean_prop_high

        # logic to reduce Qc of high DA low slope reaches
        # for i in network.index:
        #     if network.loc[i, 'Drain_Area'] >= 300 and network.loc[i, 'Slope'] <= 0.008:
        #         network.loc[i, 'Qc'] = network.loc[i, 'Qc'] / 4

        network.to_file(self.streams)

        return

    def find_Dpred(self):  # can vectorize this
        """
        Estimates median grain size for each drainage network segment based on Qc value and adds a field for
        this predicted median grain size to the attribute table
        :return:
        """

        network = gpd.read_file(self.streams)

        rho = 1000
        g = 9.81
        rho_s = 2650

        Qc_mid = network['Qc_mid']
        Qc_low = network['Qc_low']
        Qc_high = network['Qc_high']
        S = network['Slope_mid']
        w = network['w_bf']
        om_crit_star = 0.08  # fixed or make param?

        om_crit_mid = (rho * g * Qc_mid * S) / w  # this should probably be ACTIVE channel width
        om_crit_low = (rho * g * Qc_low * S) / w
        om_crit_high = (rho * g * Qc_high * S) / w

        numerator_mid = (om_crit_mid / om_crit_star) ** 2
        numerator_low = (om_crit_low / om_crit_star) ** 2
        numerator_high = (om_crit_high / om_crit_star) ** 2
        denominator = ((g * (rho - rho_s)) ** 2) * (((rho_s / rho) - 1) * g)

        Dpred_mid = ((numerator_mid / denominator) ** (1. / 3.)) * 1000.
        Dpred_low = ((numerator_low / denominator) ** (1. / 3.)) * 1000.
        Dpred_high = ((numerator_high / denominator) ** (1. / 3.)) * 1000.

        network['D_pred_mid'] = Dpred_mid  # * (-0.85 * network['f_sand'] + 1)  # this equation for reducing D50 based on sand probably isnt good enough
        network['D_pred_low'] = Dpred_low  # * (-0.85 * network['f_sand'] + 1)
        network['D_pred_high'] = Dpred_high  # * (-0.85 * network['f_sand'] + 1)

        network.to_file(self.streams)

        return
