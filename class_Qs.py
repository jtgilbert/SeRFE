# imports
import network_topology as nt
import geopandas as gpd
import pandas as pd


class BaselineQs:
    """Uses data on watershed sediment yields to attribute each segment of the input network with a value
    for Qs, representing a pre-disturbance average sediment yield. This class takes an input table with reach ids
    for watershed 'pour points' above which the table value for sediment yield (tonnes/km2/yr) applies."""

    def __init__(self, network, qs_table):
        """

        :param network: string - path to stream network shapefile with topology attributes
        :param qs_table: string - path to csv with headers 'Segment ID': segment ID for downstream reach that bounds a
        specific value for average annual sediment yield 'Qs': the associated value of sediment yield that applies
        to segments upstream of associated segment ID (tonnes/km2/yr).
        """

        self.streams = network
        self.network = gpd.read_file(network)

        table = pd.read_csv(qs_table, sep=',', header=0)
        table = table.dropna(axis='columns')

        # add da values to table and then sort by da so Qs is attributed from top of network down
        da_values = []
        for x in table.index:
            da = self.network.loc[table.loc[x, 'Segment ID'], 'Drain_Area']
            da_values.append(da)

        table['DA'] = da_values
        sortedtable = table.sort_values(by='DA')
        self.sortedtable = sortedtable.reset_index(drop=True)
        self.segids = self.sortedtable['Segment ID']
        self.values = self.sortedtable['Qs']

        self.topo = nt.TopologyTools(self.streams)

        self.qs_to_network()

    def qs_to_network(self):
        """

        :return:
        """

        self.network['Qs (t/yr)'] = -9999
        self.network['sed_rate'] = -9999

        for x in range(len(self.segids)):
            self.network.loc[self.segids[x], 'Qs (t/yr)'] = self.values[x]*self.network.loc[self.segids[x], 'Drain_Area']
            self.network.loc[self.segids[x], 'sed_rate'] = self.values[x]
            us_list = self.topo.find_all_us(self.segids[x])
            for i in us_list:
                if self.network.loc[i, 'Qs (t/yr)'] == -9999:
                    self.network.loc[i, 'Qs (t/yr)'] = self.values[x]*self.network.loc[i, 'Drain_Area']
                    self.network.loc[i, 'sed_rate'] = self.values[x]

        self.network.to_file(self.streams)

        return


class UpdatedQs:
    """

    """

    def __init__(self, network, updated_qs_table):
        """

        :param network:
        :param updated_qs_table:
        """

        self.streams = network
        self.network = gpd.read_file(self.streams)
        table = pd.read_csv(updated_qs_table, sep=',', header=0)
        table = table.dropna(axis='columns')

        # add da values to table and then sort by da so disturbance is accounted for from upstream to downstream
        da_values = []
        for x in table.index:
            da = self.network.loc[table.loc[x, 'Segment ID'], 'Drain_Area']
            da_values.append(da)

        table['DA'] = da_values
        sortedtable = table.sort_values(by='DA')
        self.sortedtable = sortedtable.reset_index(drop=True)
        self.segids = self.sortedtable['Segment ID']
        self.values = self.sortedtable['newQs']

        self.topo = nt.TopologyTools(network)

        self.network['newQs'] = -9999
        self.network['eff_DA'] = -9999

    def unaffected_reaches(self):
        """Copy baseline rate, Qs and effective DA for any stream reaches that are not downstream of any disturbance."""

        ds_list = []

        for x in range(len(self.segids)):
            ds = self.topo.find_all_ds(self.segids[x])
            ds_list.extend(ds)

        for i in self.network.index:
            if i not in ds_list:
                self.network.loc[i, 'newQs'] = self.network.loc[i, 'Qs (t/yr)']
                self.network.loc[i, 'eff_DA'] = self.network.loc[i, 'Drain_Area']

        self.network.to_file(self.streams)

        return

    def ds_reductions(self):
        """Update the newQs value downstream of disturbance until the next confluence"""

        for x in range(len(self.segids)):
            self.network.loc[self.segids[x], 'newQs'] = self.values[x]
            next_reach = self.topo.get_next_reach(self.segids[x])
            while next_reach is not None:
                if self.network.loc[next_reach, 'confluence'] == 0:
                    dist_da = self.network.loc[self.segids[x], 'Drain_Area']
                    eff_da = self.network.loc[next_reach, 'Drain_Area'] - dist_da
                    self.network.loc[next_reach, 'eff_DA'] = eff_da
                    self.network.loc[next_reach, 'newQs'] = self.values[x] + self.network.loc[next_reach, 'sed_rate'] * eff_da
                    next_reach = self.topo.get_next_reach(next_reach)
                else:
                    next_reach = None

        self.network.to_file(self.streams)

        return

    def update_below_confluences(self):

        conf_list = []

        for i in self.network.index:
            if self.network.loc[i, 'confluence'] == 1:
                if self.network.loc[i, 'newQs'] == -9999:
                    conf_list.append(i)

        while len(conf_list) > 0:
            for x in conf_list:

                us1 = self.topo.find_us_seg(x)
                us1qs = self.network.loc[us1, 'newQs']
                us1_eff_da = self.network.loc[us1, 'eff_DA']
                us2 = self.topo.find_us_seg2(x)
                us2qs = self.network.loc[us2, 'newQs']
                us2_eff_da = self.network.loc[us2, 'eff_DA']
                if us1qs != -9999 and us2qs != -9999:

                    self.network.loc[x, 'newQs'] = us1qs + us2qs
                    self.network.loc[x, 'eff_DA'] = us1_eff_da + us2_eff_da

                    next_reach = self.topo.get_next_reach(x)
                    while next_reach is not None:

                        if self.network.loc[next_reach, 'confluence'] == 0:
                            if self.network.loc[next_reach, 'newQs'] == -9999:
                                us_reach = self.topo.find_us_seg(next_reach)
                                da_prop = self.network.loc[us_reach, 'eff_DA']/self.network.loc[next_reach, 'Drain_Area']
                                self.network.loc[next_reach, 'eff_DA'] = self.network.loc[next_reach, 'Drain_Area'] * da_prop
                                qs_add = self.network.loc[us_reach, 'newQs'] - (self.network.loc[us_reach, 'eff_DA']*self.network.loc[us_reach, 'sed_rate'])
                                self.network.loc[next_reach, 'newQs'] = (self.network.loc[next_reach, 'eff_DA'] * self.network.loc[next_reach, 'sed_rate']) + qs_add
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
