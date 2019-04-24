#imports
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString


class Confinement:
    """
    Calculates confinement for each reach of an input drainage network and adds an attribute with this value
    """
    def __init__(self, network, valley, exag=0.5): # exag: proportion of channel width to add to buffer to makes sure there's intersection with valley
        """

        :param network: string - path to drainage network shapefile
        :param valley: string - path to valley bottom shapefile
        :param exag: float - a proportion (0 - 1) of the stream network width at each segment to add to the buffer
        width to ensure overlap with the valley bottom polygon. Default = 0.5
        """
        self.streams = network
        self.network = gpd.read_file(network)
        self.valley = gpd.read_file(valley)
        self.exag = exag

        # set confinement value to default nodata
        self.network['confine'] = -9999

    def calc_confinement(self, seg, buf_width):
        """
        Calculates confinement. this function is called in the confinement method.
        """

        channel = seg.buffer(buf_width)
        dif = channel.difference(self.valley.loc[0, 'geometry'])
        inters = channel.intersection(self.valley.loc[0, 'geometry'])

        if inters.type == 'MultiPolygon':
            int_coords = []
            for i in range(len(inters)):
                for j in range(len(inters[i].exterior.xy[0])):
                    int_coords_x = inters[i].exterior.xy[0][j]
                    int_coords_y = inters[i].exterior.xy[1][j]
                    int_coords.append([int_coords_x, int_coords_y])
                    # print int_coords
        elif inters.type == 'Polygon':
            int_coords = np.zeros((len(inters.exterior.xy[0]), 2))
            int_coords[:, 0] = inters.exterior.xy[0]
            int_coords[:, 1] = inters.exterior.xy[1]
        else:
            int_coords = []

        if dif.type == 'MultiPolygon':
            line_len = []
            line_coords = []
            for i in range(len(dif)):
                for j in range(len(dif[i].exterior.xy[0])):
                    dif_coords_x = dif[i].exterior.xy[0][j]
                    dif_coords_y = dif[i].exterior.xy[1][j]
                    if [dif_coords_x, dif_coords_y] in int_coords:
                        line_coords.append([dif_coords_x, dif_coords_y])
                line = LineString(line_coords[1:len(line_coords)])
                line_len.append(line.length)
        elif dif.type == 'Polygon':
            line_len = []
            line_coords = []
            for y in range(len(dif.exterior.xy[0])):
                dif_coords_x = dif.exterior.xy[0][y]
                dif_coords_y = dif.exterior.xy[1][y]
                if [dif_coords_x, dif_coords_y] in int_coords:
                    line_coords.append([dif_coords_x, dif_coords_y])
            # print 'line_coords', len(line_coords)
            line = LineString(line_coords[1:len(line_coords)])
            line_len.append(line.length)
        else:
            line_len = []

        if len(int_coords) == 0:
            return 1.  # stream network and valley bottom misaligned
        elif len(line_len) == 0:
            return 0.  # no overlap, stream is fully unconfined
        else:
            return min(1., np.sum(line_len) / (2*seg.length))

    def confinement(self):
        """
        Apply the confinement algorithm to each segment of the input drainage network
        :return: adds value for attribute 'confine' to each segment of the drainage network
        """
        for i in self.network.index:
            print i
            seg = self.network.loc[i, 'geometry']
            buf_width = (self.network.loc[i, 'w_bf']/2) + (self.network.loc[i, 'w_bf']*self.exag)

            conf_val = self.calc_confinement(seg, buf_width)
            #print seg, conf_val

            self.network.loc[i, 'confine'] = conf_val

        self.network.to_file(self.streams)

        return

    def update_width(self):

        network = gpd.read_file(self.streams)

        for i in network.index:
            if network.loc[i, 'confine'] == 1.:
                network.loc[i, 'w_flood'] = network.loc[i, 'w_bf']

        network.to_file(self.streams)

        return

    def update_area(self):

        network = gpd.read_file(self.streams)

        for i in network.index:
            if network.loc[i, 'confine'] == 1.:
                network.loc[i, 'fp_area'] = 0.

        network.to_file(self.streams)

        return
