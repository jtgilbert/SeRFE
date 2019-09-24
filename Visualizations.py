# imports
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd


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
        ax.plot(self.df.index.levels[0][:-1], series_min, linewidth=3, linestyle=':', color='blue')
        ax.plot(self.df.index.levels[0][:-1], series_mid, linewidth=3, color='blue')
        ax.plot(self.df.index.levels[0][:-1], series_max, linewidth=3, linestyle='dashed', color='blue')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.axhline(1, linestyle='dashed', color='k')
        ax.set_xlabel('Time Step', fontsize=16)
        ax.set_ylabel('CSR', fontsize=16)
        ax.set_title("Segment {0}".format(seg), fontsize=20, fontweight='bold')
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
        for x in self.df.index.levels[0]:
            val_min1 = self.df.loc[(x, seg), 'Store_tot_min']
            val_min2 = self.df.loc[(x, seg), 'Store_chan_min']
            val_mid1 = self.df.loc[(x, seg), 'Store_tot_mid']
            val_mid2 = self.df.loc[(x, seg), 'Store_chan_mid']
            val_max1 = self.df.loc[(x, seg), 'Store_tot_max']
            val_max2 = self.df.loc[(x, seg), 'Store_chan_max']
            series_min1.append(val_min1)
            series_min2.append(val_min2)
            series_mid1.append(val_mid1)
            series_mid2.append(val_mid2)
            series_max1.append(val_max1)
            series_max2.append(val_max2)
        series_min1 = series_min1[0:-1]
        series_min2 = series_min2[0:-1]
        series_mid1 = series_mid1[0:-1]
        series_mid2 = series_mid2[0:-1]
        series_max1 = series_max1[0:-1]
        series_max2 = series_max2[0:-1]

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

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.df.index.levels[0][0:-1], series_mid1, linewidth=3, color='g', label='total_storage (mid)')
        ax.plot(self.df.index.levels[0][0:-1], series_min1, linestyle=':', linewidth=3, color='g', label='min')
        ax.plot(self.df.index.levels[0][0:-1], series_max1, linestyle='dashed', linewidth=3, color='g', label='max')
        ax.plot(self.df.index.levels[0][0:-1], series_fp_mid, linewidth=3, color='c', label='floodplain (mid)')
        ax.plot(self.df.index.levels[0][0:-1], series_fp_min, linestyle=':', linewidth=3, color='c', label='min')
        ax.plot(self.df.index.levels[0][0:-1], series_fp_max, linestyle='dashed', linewidth=3, color='c', label='max')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.axhline(series_mid1[0], linestyle='dashed', color='k', label='Initial value')
        ax.set_xlabel('Time Step', fontsize=16)
        ax.set_ylabel('Storage (tonnes)', fontsize=16)
        ax.set_title("Segment {0}: Sediment Storage".format(seg), fontsize=20, fontweight='bold')
        ax.legend(fontsize=14)
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

    def delta_storage_plot(self):
        for i in self.network.index:
            s1_min = self.df.loc[(1, i), 'Store_tot_min']
            s2_min = self.df.loc[(self.df.index.levels[0][-2], i), 'Store_tot_min']
            delta_store_min = s2_min - s1_min
            self.network.loc[i, 'd_Stor_min'] = delta_store_min
            s1_mid = self.df.loc[(1, i), 'Store_tot_mid']
            s2_mid = self.df.loc[(self.df.index.levels[0][-2], i), 'Store_tot_mid']
            delta_store_mid = s2_mid - s1_mid
            self.network.loc[i, 'd_Stor_mid'] = delta_store_mid
            s1_max = self.df.loc[(1, i), 'Store_tot_max']
            s2_max = self.df.loc[(self.df.index.levels[0][-2], i), 'Store_tot_max']
            delta_store_max = s2_max - s1_max
            self.network.loc[i, 'd_Stor_max'] = delta_store_max

        self.network.to_file(self.streams)
        fig, ax = plt.subplots(figsize=(10, 7))
        self.network.plot(ax=ax, column='d_Stor_mid', cmap='RdYlBu')
        plt.show()

    def csr_integrate(self):
        for i in self.network.index:
            tot_min = []
            tot_mid = []
            tot_max = []
            for x in self.df.index.levels[0]:
                val_min = self.df.loc[(x, i), 'CSR_min'] - 1
                tot_min.append(val_min)
                val_mid = self.df.loc[(x, i), 'CSR_mid'] - 1
                tot_mid.append(val_mid)
                val_max = self.df.loc[(x, i), 'CSR_max'] - 1
                tot_max.append(val_max)

            self.network.loc[i, 'CSR_min'] = np.sum(tot_min)/365
            self.network.loc[i, 'CSR_mid'] = np.sum(tot_mid)/365
            self.network.loc[i, 'CSR_max'] = np.sum(tot_max)/365

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


inst = Visualizations('Piru/outputs/piru_out_nospinup.csv', 'Piru/outputs/piru_nospinup.shp', 'Piru/Piru_hydrographs.csv')
#inst.sum_plot('Qs')
#inst.sum_plot('Qs_out')
#inst.delta_storage_plot()
#inst.csr_integrate()
#inst.plot_csr_time_series(301)
inst.plot_storage(204)
#inst.plot_time_series(519, 'Q')
#inst.date_fig(232, 'CSR_mid', save=True)

#inst2 = Stats('/home/jordan/Documents/piru_output.csv')
#print inst2.seg_mean(17, 'CSR')
#print inst2.seg_sum(487, 'CSR')/365
