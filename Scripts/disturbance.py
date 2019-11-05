# imports
import geopandas as gpd
import network_topology as nt
import numpy as np


class Disturbances:
    """Update stream network with relevant information regarding watershed disturbances
    prior to running simulations."""

    def __init__(self, network):
        """
        :param network: drainage network shapefile
        """

        self.streams = network
        self.network = gpd.read_file(network)

        self.topo = nt.TopologyTools(network)

        # add an 'effective drainage area' field, default to actual DA
        if 'eff_DA' in self.network.columns:
            pass
        else:
            for i in self.network.index:
                self.network.loc[i, 'eff_DA'] = self.network.loc[i, 'Drain_Area']

        # add a denudation rate field, default to -9999
        if 'denude' in self.network.columns:
            pass
        else:
            for i in self.network.index:
                self.network.loc[i, 'denude'] = -9999

        # add a disturbance start time field
        if 'dist_start' in self.network.columns:
            pass
        else:
            for i in self.network.index:
                self.network.loc[i, 'dist_start'] = -9999

        # add a disturbance end time field
        if 'dist_end' in self.network.columns:
            pass
        else:
            for i in self.network.index:
                self.network.loc[i, 'dist_end'] = -9999

        # add a disturbance gamma shape and scale
        if 'dist_g_sh' in self.network.columns:
            pass
        else:
            for i in self.network.index:
                self.network.loc[i, 'dist_g_sh'] = -9999
                self.network.loc[i, 'dist_g_sc'] = -9999

        self.network.to_file(self.streams)

    def add_disturbance(self, segid, new_da=False, dist_start=None, dist_end=None, new_denude=None):
        """
        Run separately for things that change effective da (e.g. dams) and
        things that increase sedimentation (e.g. fire)
        :param segid:
        :param new_da:
        :param new_sed:
        :return:
        """

        if new_da:
            for x in range(len(segid)):
                da = self.network.loc[segid[x], 'Drain_Area']
                ds_segs = self.topo.find_all_ds(segid[x])
                for y in range(len(ds_segs)):
                    self.network.loc[ds_segs[y], 'eff_DA'] = self.network.loc[ds_segs[y], 'eff_DA'] - da

        if new_denude is not None:
            for x in range(len(segid)):
                if len(dist_start) > 1:
                    self.network.loc[segid[x], 'dist_start'] = dist_start[x]
                else:
                    self.network.loc[segid[x], 'dist_start'] = dist_start[0]
                if len(dist_end) > 1:
                    self.network.loc[segid[x], 'dist_end'] = dist_end[x]
                else:
                    self.network.loc[segid[x], 'dist_end'] = dist_end[0]
                if len(new_denude) > 1:
                    self.network.loc[segid[x], 'dist_g_sh'] = new_denude[x][0]
                    self.network.loc[segid[x], 'dist_g_sc'] = new_denude[x][1]
                else:
                    self.network.loc[segid[x], 'dist_g_sh'] = new_denude[0][0]
                    self.network.loc[segid[x], 'dist_g_sc'] = new_denude[0][1]

        self.network.to_file(self.streams)

        return

    def update_direct_da(self):
        """

        :return:
        """

        # add directly contributing DA to each network segment
        for i in self.network.index:
            print 'segment ' + str(i)
            us_seg = self.topo.find_us_seg(i)
            us_seg2 = self.topo.find_us_seg2(i)
            if us_seg is not None:
                da1 = self.network.loc[us_seg, 'eff_DA']
                if us_seg2 is not None:
                    da2 = self.network.loc[us_seg2, 'eff_DA']
                    us_da = da1 + da2
                    dda = self.network.loc[i, 'eff_DA'] - us_da
                else:
                    dda = self.network.loc[i, 'eff_DA'] - da1

            else:
                dda = self.network.loc[i, 'eff_DA']

            self.network.loc[i, 'direct_DA'] = dda

            self.network.to_file(self.streams)

        return
