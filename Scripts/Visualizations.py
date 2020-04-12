# imports
import matplotlib.pyplot as plt
from matplotlib import ticker
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import gmean
import scipy as sp


class Visualizations:

    def __init__(self, df, network, hyd):
        self.df = pd.read_csv(df, index_col=[0, 1])
        self.network = gpd.read_file(network)
        self.streams = network
        self.hyd = pd.read_csv(hyd, index_col='Gage')

    def plot_csr_time_series(self, seg):
        series_min = []
        series_mid = []
        series_max = []
        for x in self.df.index.levels[0]:
            val_min = self.df.loc[(x, seg), 'CSR_min']
            val_mid = self.df.loc[(x, seg), 'CSR_mid']
            val_max = self.df.loc[(x, seg), 'CSR_max']
            series_min.append(val_min)
            series_mid.append(val_mid)
            series_max.append(val_max)

        series_min = series_min[0:-1]
        series_mid = series_mid[0:-1]
        series_max = series_max[0:-1]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.df.index.levels[0][:-1], series_min, linewidth=1, linestyle=':', color='blue', label='Lower tranpsort')
        ax.plot(self.df.index.levels[0][:-1], series_mid, linewidth=3, color='blue', label='CSR')
        ax.plot(self.df.index.levels[0][:-1], series_max, linewidth=1, linestyle='dashdot', color='blue', label='Higher transport')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.axhline(1, linestyle='dashed', color='k')
        ax.axhline(np.max(series_mid), linestyle=':', color='k')
        ax.set_xlabel('Time Step', fontsize=20)
        ax.set_ylabel('CSR', fontsize=20)
        ax.set_title("Segment {0}".format(seg), fontsize=20, fontweight='bold')
        ax.legend(fontsize=16)
        ax.set_yscale('log')
        plt.minorticks_on()
        plt.text(50, (np.max(series_mid))-5, 'CSR max: {0:.2f}'.format(np.max(series_mid)), fontsize=16)
        plt.show()

        return

    def plot_time_series(self, seg, att):
        series = []
        for x in self.df.index.levels[0]:
            val = self.df.loc[(x, seg), att]
            series.append(val)
        series = series[0:-1]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.df.index.levels[0][0:-1], series, linewidth=5, color='blue')
        ax.set_xlabel('Time Step', fontsize=16)
        ax.set_ylabel(str(att), fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        if att == 'Q':
            att = 'Discharge'
        ax.set_title("Segment {0}: {1}".format(seg, att), fontsize=20, fontweight='bold')
        plt.show()

        return

    def plot_storage(self, seg):
        series_min1 = []
        series_min2 = []
        series_mid1 = []
        series_mid2 = []
        series_max1 = []
        series_max2 = []
        q_series = []
        for x in self.df.index.levels[0]:
            val_min1 = self.df.loc[(x, seg), 'Store_tot_min']
            val_min2 = self.df.loc[(x, seg), 'Store_chan_min']
            val_mid1 = self.df.loc[(x, seg), 'Store_tot_mid']
            val_mid2 = self.df.loc[(x, seg), 'Store_chan_mid']
            val_max1 = self.df.loc[(x, seg), 'Store_tot_max']
            val_max2 = self.df.loc[(x, seg), 'Store_chan_max']
            q_val = self.df.loc[(x, seg), 'Q']
            series_min1.append(val_min1)
            series_min2.append(val_min2)
            series_mid1.append(val_mid1)
            series_mid2.append(val_mid2)
            series_max1.append(val_max1)
            series_max2.append(val_max2)
            q_series.append(q_val)
        series_min1 = series_min1[0:-1]
        series_min2 = series_min2[0:-1]
        series_mid1 = series_mid1[0:-1]
        series_mid2 = series_mid2[0:-1]
        series_max1 = series_max1[0:-1]
        series_max2 = series_max2[0:-1]
        q_series = q_series[0:-1]

        series_fp_min = []
        series_fp_mid = []
        series_fp_max = []
        for i in range(len(series_min1)):
            val_fp_min = series_min1[i] - series_min2[i]
            val_fp_mid = series_mid1[i] - series_mid2[i]
            val_fp_max = series_max1[i] - series_max2[i]
            series_fp_min.append(val_fp_min)
            series_fp_mid.append(val_fp_mid)
            series_fp_max.append(val_fp_max)

        fig, ax = plt.subplots(2, 1, figsize=(12,20))
        ax[0].plot(self.df.index.levels[0][0:-1], series_mid1, linewidth=3, color='g', label='Total storage')
        ax[0].plot(self.df.index.levels[0][0:-1], series_fp_mid, linewidth=3, color='c', label='Floodplain storage')
        ax[0].plot(self.df.index.levels[0][0:-1], series_min1, linestyle=':', linewidth=1, color='g', label='Lower transport')
        ax[0].plot(self.df.index.levels[0][0:-1], series_max1, linestyle='dashdot', linewidth=1, color='g', label='Higher transport')
        ax[0].plot(self.df.index.levels[0][0:-1], series_fp_min, linestyle=':', linewidth=1, color='c', label='_nolabel_')
        ax[0].plot(self.df.index.levels[0][0:-1], series_fp_max, linestyle='dashdot', linewidth=1, color='c', label='_nolabel_')
        ax[0].tick_params(axis='both', which='major', labelsize=16)
        ax[0].axhline(series_mid1[0], linestyle='dashed', color='k', label='Initial value')
        ax[0].set_xlabel('Time Step', fontsize=20)
        ax[0].set_ylabel('Storage (tonnes)', fontsize=20)
        ax[0].set_title("Segment {0}: Sediment Storage".format(seg), fontsize=20, fontweight='bold')
        ax[0].legend(fontsize=16)
        ax[1].plot(self.df.index.levels[0][0:-1], q_series, linewidth=5, color='k')
        ax[1].set_xlabel('Time Step', fontsize=16)
        ax[1].set_ylabel('Discharge (cms)', fontsize=16)
        ax[1].tick_params(axis='both', which='major', labelsize=16)
        #ax[1].set_title("Segment {0}: Discharge".format(seg), fontsize=20, fontweight='bold')
        fig.tight_layout(pad=8)
        plt.show()

        return

    def date_fig(self, time, att, save=True):  # outdated need to update

        att_name = att + str(time)
        for i in self.network.index:
            self.network.loc[i, att_name] = self.df.loc[(time, i), att]

        if save:
            self.network.to_file(self.streams)

        fig, ax = plt.subplots(figsize=(10,7))
        self.network.plot(ax=ax, column=att_name, cmap='RdYlBu')
        plt.show()

        return

    def sum_plot(self, att):
        att_name = att + '_total'
        for i in self.network.index:
            tot = []
            for x in self.df.index.levels[0]:
                val = self.df.loc[(x, i), att]
                tot.append(val)

            self.network.loc[i, att_name] = sum(tot)

        self.network.to_file(self.streams)
        fig, ax = plt.subplots(figsize=(10,7))
        self.network.plot(ax=ax, column=att_name, cmap='RdYlBu')
        plt.show()

    def mean_plot(self, att):
        att_name = att + '_mean'
        for i in self.network.index:
            tot = []
            for x in self.df.index.levels[0]:
                val = self.df.loc[(x, i), att]
                tot.append(val)

            self.network.loc[i, att_name] = np.mean(tot)

        self.network.to_file(self.streams)
        fig, ax = plt.subplots(figsize=(10,7))
        self.network.plot(ax=ax, column=att_name, cmap='RdYlBu')
        plt.show()

    def delta_storage_plot(self, start, end):
        for i in self.network.index:
            s1_min = self.df.loc[(start, i), 'Store_tot_min']
            s2_min = self.df.loc[(end, i), 'Store_tot_min']
            self.network.loc[i, 'd_Stor_min'] = s2_min - s1_min
            s1_mid = self.df.loc[(start, i), 'Store_tot_mid']
            s2_mid = self.df.loc[(end, i), 'Store_tot_mid']
            self.network.loc[i, 'd_Stor_mid'] = s2_mid - s1_mid
            s1_max = self.df.loc[(start, i), 'Store_tot_max']
            s2_max = self.df.loc[(end, i), 'Store_tot_max']
            self.network.loc[i, 'd_Stor_max'] = s2_max - s1_max

        self.network.to_file(self.streams)
        fig, ax = plt.subplots(figsize=(10, 7))
        self.network.plot(ax=ax, column='d_Stor_mid', cmap='RdYlBu')
        plt.show()

    def d_storage_atts(self, start, end):
        for i in self.network.index:
            fp_start = self.df.loc[(start, i), 'Store_tot_mid']-self.df.loc[(start, i), 'Store_chan_mid']
            fp_end = self.df.loc[(end, i), 'Store_tot_mid']-self.df.loc[(end, i), 'Store_chan_mid']
            fp_start_min = self.df.loc[(start, i), 'Store_tot_min'] - self.df.loc[(start, i), 'Store_chan_min']
            fp_end_min = self.df.loc[(end, i), 'Store_tot_min'] - self.df.loc[(end, i), 'Store_chan_min']
            fp_start_max = self.df.loc[(start, i), 'Store_tot_max'] - self.df.loc[(start, i), 'Store_chan_max']
            fp_end_max = self.df.loc[(end, i), 'Store_tot_max'] - self.df.loc[(end, i), 'Store_chan_max']
            self.network.loc[i, 'd_fp_min'] = fp_end_min - fp_start_min
            self.network.loc[i, 'd_fp_mid'] = fp_end - fp_start
            self.network.loc[i, 'd_fp_max'] = fp_end_max - fp_start_max
            self.network.loc[i, 'd_ch_min'] = self.df.loc[(end, i), 'Store_chan_min'] - self.df.loc[(start, i), 'Store_chan_min']
            self.network.loc[i, 'd_ch_mid'] = self.df.loc[(end, i), 'Store_chan_mid'] - self.df.loc[(start, i), 'Store_chan_mid']
            self.network.loc[i, 'd_ch_max'] = self.df.loc[(end, i), 'Store_chan_max'] - self.df.loc[(start, i), 'Store_chan_max']
        self.network.to_file(self.streams)

    def csr_integrate(self, start, end):
        for i in self.network.index:
            tot_min = []
            tot_mid = []
            tot_max = []
            #for x in self.df.index.levels[0]:
            for x in range(start, end+1):
                tot_min.append(self.df.loc[(x, i), 'CSR_min'])
                tot_mid.append(self.df.loc[(x, i), 'CSR_mid'])
                tot_max.append(self.df.loc[(x, i), 'CSR_max'])

            # self.network.loc[i, 'CSR_min'] = np.sum(tot_min)/len(self.df.index.levels[0])
            # self.network.loc[i, 'CSR_mid'] = np.sum(tot_mid)/len(self.df.index.levels[0])
            # self.network.loc[i, 'CSR_max'] = np.sum(tot_max)/len(self.df.index.levels[0])
            self.network.loc[i, 'CSR_min'] = gmean(tot_min)
            self.network.loc[i, 'CSR_mid'] = gmean(tot_mid)
            self.network.loc[i, 'CSR_max'] = gmean(tot_max)
            self.network.loc[i, 'CSRmin_max'] = max(tot_min)
            self.network.loc[i, 'CSRmid_max'] = max(tot_mid)
            self.network.loc[i, 'CSRmax_max'] = max(tot_max)
            self.network.loc[i, 'CSRmin_std'] = sp.exp(sp.sqrt(sp.sum([sp.log(x / gmean(tot_min)) ** 2 for x in tot_min]) / len(tot_min)))
            self.network.loc[i, 'CSRmid_std'] = sp.exp(sp.sqrt(sp.sum([sp.log(x / gmean(tot_mid)) ** 2 for x in tot_mid]) / len(tot_mid)))
            self.network.loc[i, 'CSRmax_std'] = sp.exp(sp.sqrt(sp.sum([sp.log(x / gmean(tot_max)) ** 2 for x in tot_max]) / len(tot_max)))

        self.network.to_file(self.streams)

    def csr_integrate2(self, start, end):
        for i in self.network.index:
            tot = []
            #for x in self.df.index.levels[0]:
            for x in range(start, end+1):
                tot.append(self.df.loc[(x, i), 'us_CSR'])

            # self.network.loc[i, 'CSR_min'] = np.sum(tot_min)/len(self.df.index.levels[0])
            # self.network.loc[i, 'CSR_mid'] = np.sum(tot_mid)/len(self.df.index.levels[0])
            # self.network.loc[i, 'CSR_max'] = np.sum(tot_max)/len(self.df.index.levels[0])
            self.network.loc[i, 'usCSR'] = gmean(tot)

            self.network.loc[i, 'usCSR_max'] = max(tot)

            self.network.loc[i, 'usCSR_std'] = sp.exp(sp.sqrt(sp.sum([sp.log(x / gmean(tot)) ** 2 for x in tot]) / len(tot)))

        self.network.to_file(self.streams)

    def integrate(self):
        for i in self.network.index:
            tot=[]
            for x in self.df.index.levels[0]:
                tot.append(self.df.loc[(x,i), 'CSR_mid'])

            self.network.loc[i, 'CSRmid_sum']=np.sum(tot)

        self.network.to_file(self.streams)


class Stats:

    def __init__(self, df):
        self.df = pd.read_csv(df, index_col=[0, 1])

    def seg_sum(self, seg, att='Qs_out'):
        series = []
        for x in self.df.index.levels[0]:
            val = self.df.loc[(x, seg), att]
            series.append(val)

        return np.sum(series)

    def seg_mean(self, seg, att):
        series = []
        for x in self.df.index.levels[0]:
            val = self.df.loc[(x, seg), att]
            series.append(val)

        return np.mean(series)

