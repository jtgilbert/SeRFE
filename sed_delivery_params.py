# imports
import geopandas as gpd
import rasterio
from shapely.geometry import LineString
from scipy.signal import convolve2d
import numpy as np
from rasterstats import zonal_stats


class SedDeliveryParams:

    def __init__(self, dem, slope_out, network, neighborhood, g_min, g_max, g_scale):
        self.dem = dem
        self.slope_out = slope_out
        self.dn = gpd.read_file(network)
        self.streams = network
        self.neighborhood = neighborhood
        self.g_min = g_min
        self.g_max = g_max
        self.g_scale = g_scale

        self.slope()
        self.get_gamma_vals()

    def slope(self):
        """
        Finds the slope using partial derivative method
        :param dem: path to a digital elevation raster
        :return: a slope raster saved to location of dem
        """
        with rasterio.open(self.dem, 'r') as src:
            meta = src.profile
            dtype = src.dtypes[0]
            arr = src.read()[0, :, :]

            xres = src.res[0]
            yres = src.res[1]

        x = np.array([[-1 / (8 * xres), 0, 1 / (8 * xres)],
                      [-2 / (8 * xres), 0, 2 / (8 * xres)],
                      [-1 / (8 * xres), 0, 1 / (8 * xres)]])
        y = np.array([[1 / (8 * yres), 2 / (8 * yres), 1 / (8 * yres)],
                      [0, 0, 0],
                      [-1 / (8 * yres), -2 / (8 * yres), -1 / (8 * yres)]])

        x_grad = convolve2d(arr, x, mode='same', boundary='fill', fillvalue=1)
        y_grad = convolve2d(arr, y, mode='same', boundary='fill', fillvalue=1)
        slope = np.arctan(np.sqrt(x_grad ** 2 + y_grad ** 2)) * (180. / np.pi)
        slope = slope.astype(dtype)

        with rasterio.open(self.slope_out, 'w', **meta) as dst:
            dst.write(slope, 1)

        return

    def get_gamma_vals(self):
        ave_slope = []

        for i in self.dn.index:
            seg = self.dn.loc[i]
            geom = seg['geometry']

            print i

            ept1 = (geom.boundary[0].xy[0][0], geom.boundary[0].xy[1][0])
            ept2 = (geom.boundary[1].xy[0][0], geom.boundary[1].xy[1][0])
            line = LineString([ept1, ept2])

            buf = line.buffer(self.neighborhood, cap_style=2)

            zs = zonal_stats(buf, self.slope_out, stats='mean')
            mean = zs[0].get('mean')

            ave_slope.append(mean)

        l_slope = (self.g_max - self.g_min)/(max(ave_slope) - min(ave_slope))
        g_shape = []
        for i in range(len(ave_slope)):
            g = ave_slope[i] * l_slope + self.g_min
            g_shape.append(g)

        self.dn['g_shape'] = g_shape
        self.dn['g_scale'] = self.g_scale

        self.dn.to_file(self.streams)

        return


# # # run get sediment delivery parameters class
wd = 'data/'
dem = wd + 'DEM_10m_Piru.tif'
slope_out = wd + 'slope.tif'
network = wd + 'Piru_network.shp'
neighborhood = 700
g_min = 0.5
g_max = 4
g_scale = 0.3

SedDeliveryParams(dem, slope_out, network, neighborhood, g_min, g_max, g_scale)
