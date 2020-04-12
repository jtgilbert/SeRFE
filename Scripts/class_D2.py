import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import os
from .network_topology import TopologyTools
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
        :param reach_ids: the drainage network reach ids associated with the locations of grain size sampling; the
        order should be the same as the order of the list of grain size measurement csv files.
        :param width_table: the width table (csv) - column 1 header: 'DA', column 2 header: 'Q', column 3 header: 'w'.
        """

        self.streams = network
        self.network = gpd.read_file(network)
        self.grain_size = grain_size
        self.reach_ids = reach_ids
        self.width_table = width_table
        self.topo = TopologyTools(self.streams)
        self.width = []
        self.slope = []
        self.Q2 = []

        if len(self.grain_size) != len(self.reach_ids):
            raise Exception('There are a different number of grain size tables and associated reach IDs')

        # add a 'bankfull' width attribute
        w_model = self.get_width_model()
        self.add_w_bf(w_model)

        for i in self.reach_ids:
            self.width.append(self.network.loc[i, 'w_bf'])
            self.slope.append(self.network.loc[i, 'Slope_mid'])
            self.Q2.append(self.network.loc[i, 'Q2 (cms)'])

        self.min_props = []
        self.mid_props = []
        self.max_props = []

        for x in range(len(self.reach_ids)):
            qc_mid, qc_low, qc_high = self.find_qc(self.grain_size[x], self.width[x], self.slope[x])
            self.min_props.append(qc_low/self.Q2[x])
            self.mid_props.append(qc_mid/self.Q2[x])
            self.max_props.append(qc_high/self.Q2[x])

        self.interpolate_qc()
        self.find_dpred()

    def get_width_model(self):

        table = pd.read_csv(self.width_table, sep=',', header=0)
        table = table.dropna(axis='columns')
        table['DA'] = np.log(table['DA'])
        table['Q'] = np.sqrt(table['Q'])

        # width regression
        regr = linear_model.LinearRegression()
        regr.fit(table[['DA', 'Q']], table['w'])
        rsq = regr.score(table[['DA', 'Q']], table['w'])
        if rsq < 0.5:
            print('R-squared is less than 0.5, poor width model fit')

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

    def find_qc(self, grain_size, width, slope):
        # set up constants
        om_crit_star = 0.1
        g = 9.81
        rho = 1000
        rho_s = 2650

        # read in grain size array
        D_array = pd.read_csv(grain_size, delimiter=',', header=0)

        # produce percent finer grain size plot
        plotname = os.path.basename(grain_size)[0:-4] + "_plot.png"
        plotpath = os.path.dirname(grain_size)
        dsort = D_array.sort_values(by=['D'])
        rank = np.arange(1, len(dsort) + 1, 1)
        dsort['rank'] = rank
        dsort['%finer'] = (dsort['rank'] / len(dsort)) * 100

        fig, ax = plt.subplots()
        ax.plot(dsort['D'], dsort['%finer'], color='red', label='_nolegend_')
        ax.axvline(dsort.median()[0], linestyle='dashed', color='k',
                   label='Median = ' + '{0:.2f}'.format(dsort.median()[0]))
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
        medgs = medgr / 1000.

        Qc_values = []

        for i in range(len(medgs)):
            numerator = om_crit_star * width * (rho_s - rho) * (
                        (((rho_s / rho) - 1.) * g * (medgs[i] ** 3.)) ** 0.5)
            denominator = rho * slope
            Qc = numerator / denominator

            Qc_values.append(max(Qc, 0.015))
        Qc_values = np.asarray(Qc_values)

        meanQc = np.mean(Qc_values)
        lowQc = max((meanQc - (1.64 * np.std(Qc_values))), 0.015)  # script currently hardcoded so that low Qc is a minimum of 1 cfs
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
        fig.savefig(plotpath + "/" + os.path.basename(grain_size)[0:-4] + "_Qc_plot.png")

        return meanQc, lowQc, highQc

    def interpolate_qc(self):

        network = gpd.read_file(self.streams)

        network['Qc_low'] = -9999
        network['Qc_mid'] = -9999
        network['Qc_high'] = -9999

        for i in range(len(self.reach_ids)):
            print('measure reach: ' + str(self.reach_ids[i]))
            us_segs = self.topo.find_all_us(self.reach_ids[i])
            if len([x for x in self.reach_ids if x in us_segs]) == 0:  # if there's no measurements upstream
                for seg in us_segs:  # set all upstream segments Qc/Q2 equal to measured Qc/Q2
                    print ('us reach: ' + str(seg))
                    network.loc[seg, 'Qc_low'] = max(self.min_props[i]*network.loc[seg, 'Q2 (cms)'], 0.015)
                    network.loc[seg, 'Qc_mid'] = self.mid_props[i]*network.loc[seg, 'Q2 (cms)']
                    network.loc[seg, 'Qc_high'] = self.max_props[i]*network.loc[seg, 'Q2 (cms)']

            # set the measurement reaches Qc
            network.loc[self.reach_ids[i], 'Qc_low'] = max(self.min_props[i] * network.loc[self.reach_ids[i], 'Q2 (cms)'], 0.015)
            network.loc[self.reach_ids[i], 'Qc_mid'] = self.mid_props[i] * network.loc[self.reach_ids[i], 'Q2 (cms)']
            network.loc[self.reach_ids[i], 'Qc_high'] = self.max_props[i] * network.loc[self.reach_ids[i], 'Q2 (cms)']

            # set each segment downstream to next confluence
            seg = self.topo.get_next_reach(self.reach_ids[i])
            while seg is not None:
                print('ds reach: ' + str(seg))
                if network.loc[seg, 'confluence'] == 0:
                    network.loc[seg, 'Qc_low'] = max(self.min_props[i] * network.loc[seg, 'Q2 (cms)'], 0.015)
                    network.loc[seg, 'Qc_mid'] = self.mid_props[i] * network.loc[seg, 'Q2 (cms)']
                    network.loc[seg, 'Qc_high'] = self.max_props[i] * network.loc[seg, 'Q2 (cms)']

                    next_reach = self.topo.get_next_reach(seg)
                    seg = next_reach
                else:
                    seg = None

        # Fill in Qc on first order streams with no measurements
        print('making sure all 1st orders are attributed')
        seg = self.topo.seg_id_from_rid('1.1')
        while seg is not None:
            if network.loc[seg, 'Qc_mid'] == -9999:
                # get the slope of all measured reaches, find most similar slope and use its qc/q2 relationship
                s = network.loc[seg, 'Slope_mid']
                slope_dif = s-self.slope
                s_dif = [abs(ele) for ele in slope_dif]
                for x in range(len(self.slope)):
                    if abs(s-self.slope[x]) == min(s_dif):
                        rat_low = network.loc[self.reach_ids[x], 'Qc_low'] / network.loc[self.reach_ids[x], 'Q2 (cms)']
                        rat_mid = network.loc[self.reach_ids[x], 'Qc_mid'] / network.loc[self.reach_ids[x], 'Q2 (cms)']
                        rat_high = network.loc[self.reach_ids[x], 'Qc_high'] / network.loc[self.reach_ids[x], 'Q2 (cms)']
                        network.loc[seg, 'Qc_low'] = max(rat_low * network.loc[seg, 'Q2 (cms)'], 0.015)
                        network.loc[seg, 'Qc_mid'] = rat_mid * network.loc[seg, 'Q2 (cms)']
                        network.loc[seg, 'Qc_high'] = rat_high * network.loc[seg, 'Q2 (cms)']
                next_reach = self.topo.get_next_reach(seg)
                if next_reach is not None:
                    seg = next_reach
                else:
                    next_reach = self.topo.get_next_chain(seg)
                    seg = next_reach
            else:
                next_reach = self.topo.get_next_chain(seg)
                seg = next_reach


        # fill in Qc downstream of confluences using the minimum of the two upstream segment ratios.
        print('below confluences')
        conf_list = network[network['confluence'] == 1].index
        da_vals = [self.network.loc[i, 'Drain_Area'] for i in conf_list]

        sort = np.argsort(da_vals)
        conf_list = [conf_list[i] for i in sort]
        print(conf_list)

        while len(conf_list) > 0:
            for x in conf_list:
                if network.loc[x, 'Qc_mid'] == -9999:
                    print('confluence :', x)
                    us1 = self.topo.find_us_seg(x)
                    us2 = self.topo.find_us_seg2(x)
                    # if both segments upstream of given segment have Qc attributes, then attribute segment
                    if network.loc[us1, 'Qc_mid'] != -9999 and network.loc[us2, 'Qc_mid'] != -9999:
                        ratio1 = network.loc[us1, 'Qc_mid']/network.loc[us1, 'Q2 (cms)']
                        ratio2 = network.loc[us2, 'Qc_mid']/network.loc[us2, 'Q2 (cms)']
                        if ratio1 < ratio2:
                            rat_low = network.loc[us1, 'Qc_low']/network.loc[us1, 'Q2 (cms)']
                            rat_mid = network.loc[us1, 'Qc_mid']/network.loc[us1, 'Q2 (cms)']
                            rat_high = network.loc[us1, 'Qc_high']/network.loc[us1, 'Q2 (cms)']
                        else:
                            rat_low = network.loc[us2, 'Qc_low']/network.loc[us2, 'Q2 (cms)']
                            rat_mid = network.loc[us2, 'Qc_mid']/network.loc[us2, 'Q2 (cms)']
                            rat_high = network.loc[us2, 'Qc_high']/network.loc[us2, 'Q2 (cms)']

                        network.loc[x, 'Qc_low'] = max(rat_low * network.loc[x, 'Q2 (cms)'], 0.015)
                        network.loc[x, 'Qc_mid'] = rat_mid * network.loc[x, 'Q2 (cms)']
                        network.loc[x, 'Qc_high'] = rat_high * network.loc[x, 'Q2 (cms)']

                        # work way downstream to next confluence
                        seg = self.topo.get_next_reach(x)
                        while seg is not None:
                            if network.loc[seg, 'confluence'] == 0:
                                if network.loc[seg, 'Qc_mid'] == -9999:
                                    print("reach: ", seg)
                                    network.loc[seg, 'Qc_low'] = max(rat_low * network.loc[seg, 'Q2 (cms)'], 0.015)
                                    network.loc[seg, 'Qc_mid'] = rat_mid * network.loc[seg, 'Q2 (cms)']
                                    network.loc[seg, 'Qc_high'] = rat_high * network.loc[seg, 'Q2 (cms)']

                                    next_reach = self.topo.get_next_reach(seg)
                                    seg = next_reach
                                else:
                                    seg = None
                            else:
                                seg = None
                        conf_list.remove(x)
                    else:
                        pass
                else:
                    conf_list.remove(x)

        network.to_file(self.streams)

        return

    def find_dpred(self):

        network = gpd.read_file(self.streams)

        rho = 1000
        g = 9.81
        rho_s = 2650

        Qc_mid = network['Qc_mid']
        Qc_low = network['Qc_low']
        Qc_high = network['Qc_high']
        S = network['Slope_mid']
        w = network['w_bf']
        om_crit_star = 0.1  # fixed or make param?

        om_crit_mid = (rho * g * Qc_mid * S) / w
        om_crit_low = (rho * g * Qc_low * S) / w
        om_crit_high = (rho * g * Qc_high * S) / w

        numerator_mid = (om_crit_mid / om_crit_star) ** 2
        numerator_low = (om_crit_low / om_crit_star) ** 2
        numerator_high = (om_crit_high / om_crit_star) ** 2
        denominator = ((g * (rho - rho_s)) ** 2) * (((rho_s / rho) - 1) * g)

        Dpred_mid = ((numerator_mid / denominator) ** (1. / 3.)) * 1000.
        Dpred_low = ((numerator_low / denominator) ** (1. / 3.)) * 1000.
        Dpred_high = ((numerator_high / denominator) ** (1. / 3.)) * 1000.

        network['D_pred_mid'] = Dpred_mid
        network['D_pred_low'] = Dpred_low
        network['D_pred_high'] = Dpred_high

        network.to_file(self.streams)

        return