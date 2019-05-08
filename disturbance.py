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

        # add a denudation rate field, default to 0
        if 'denude' in self.network.columns:
            pass
        else:
            for i in self.network.index:
                self.network.loc[i, 'denude'] = -9999

    def add_disturbance(self, segid, new_da=None, new_denude=None):
        """
        Run separately for things that change effective da (e.g. dams) and
        things that increase sedimentation (e.g. fire)
        :param segid:
        :param new_da:
        :param new_sed:
        :return:
        """

        if new_da is not None:
            for x in range(len(segid)):
                da = self.network.loc[segid[x], 'Drain_Area']
                ds_segs = self.topo.find_all_ds(segid[x])
                for y in range(len(ds_segs)):
                    self.network.loc[ds_segs[y], 'eff_DA'] = self.network.loc[ds_segs[y], 'eff_DA'] - da

        if new_denude is not None:
            for x in range(len(segid)):
                self.network.loc[segid[x], 'denude'] = np.random.gamma(new_denude[0], new_denude[1])

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


wd = 'SP/'
network = wd + 'SP_network_500m.shp'
segid = [11, 12, 13, 14, 15, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         43, 44, 45, 46, 47, 48, 49,
         50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
         74, 75, 76, 77, 78, 79, 80,
         81, 82, 83]
#newda = [0, 0]
new_denude = [5, 0.3]  # new gamma shape and scale

inst = Disturbances(network)
inst.add_disturbance(segid=segid, new_denude=new_denude)
inst.update_direct_da()
