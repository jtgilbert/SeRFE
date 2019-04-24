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

    def plot_csr_time_series(self, seg, att='CSR'):
        series = []
        for x in self.df.index.levels[0]:
            val = self.df.loc[(x, seg), att]
            series.append(val)
        series = series[0:-1]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.df.index.levels[0][:-1], series, linewidth=5, color='blue')
        ax.axhline(1, linestyle='dashed', color='k')
        ax.set_xlabel('Time Step', fontsize='large')
        ax.set_ylabel('CSR', fontsize='x-large')
        ax.set_title("Segment {0}".format(seg), fontsize='x-large', fontweight='bold')
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
        #ax.axhline(1, linestyle='dashed', color='k')
        ax.set_xlabel('Time Step', fontsize='large')
        ax.set_ylabel(str(att), fontsize='x-large')
        ax.set_title("Segment {0}: {1}".format(seg, att), fontsize='x-large', fontweight='bold')
        plt.show()

        return

    def plot_storage(self, seg, hyd_gage, att='Store_tot'):
        series = []
        for x in self.df.index.levels[0]:
            val = self.df.loc[(x, seg), att]
            series.append(val)
        series = series[0:-1]

        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        ax[0].plot(self.df.index.levels[0][0:-1], series, linewidth=5, color='g', label='_nolegend_')
        ax[0].axhline(series[0], linestyle='dashed', color='k', label='Initial value')
        ax[0].set_xlabel('Time Step', fontsize=14)
        ax[0].set_ylabel('Storage', fontsize=14)
        ax[0].set_title("Segment {0}: Total Sediment Storage".format(seg), fontsize='x-large', fontweight='bold')
        ax[0].legend()
        ax[1].plot(self.df.index.levels[0][0:-1], self.hyd.loc[hyd_gage][4:-1], color='k', linewidth=3)
        ax[1].set_xlabel('Time Step', fontsize=14)
        ax[1].set_ylabel('Q (cms) at nearest gage', fontsize=14)
        plt.show()

        return

    def date_fig(self, time, att='S*'):

        for i in self.network.index:
            self.network.loc[i, 'S*'] = self.df.loc[(time, i), att]

        fig, ax = plt.subplots(figsize=(10,7))
        self.network.plot(ax=ax, column='S*', cmap='RdYlBu')
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
            s1 = self.df.loc[(1, i), 'Store_tot']
            s2 = self.df.loc[(364, i), 'Store_tot']
            delta_store = s2 - s1
            self.network.loc[i, 'd_Store'] = delta_store

        self.network.to_file(self.streams)
        fig, ax = plt.subplots(figsize=(10, 7))
        self.network.plot(ax=ax, column='d_Store', cmap='RdYlBu')
        plt.show()

    def csr_integrate(self):
        for i in self.network.index:
            tot = []
            for x in self.df.index.levels[0]:
                val = self.df.loc[(x, i), 'CSR'] - 1
                tot.append(val)
            self.network.loc[i, 'CSR_sum'] = np.sum(tot)

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


inst = Visualizations('/home/jordan/Documents/piru_output_n4tl16.csv', 'data/Piru_network.shp', 'data/Piru_hydrographs.csv')
#inst.sum_plot('Qs')
#inst.sum_plot('Qs_out')
#inst.delta_storage_plot()
#inst.csr_integrate()
inst.plot_csr_time_series(324)
#inst.plot_storage(351, 'Wheeler')
#inst.plot_time_series(597, 'Q')

#inst2 = Stats('/home/jordan/Documents/piru_output.csv')
#print inst2.seg_mean(17, 'CSR')
#print inst2.seg_sum(487, 'CSR')/365
