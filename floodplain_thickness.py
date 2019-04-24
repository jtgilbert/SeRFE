# imports
import geopandas as gpd
from rasterstats import zonal_stats


def est_fp_thickness(dn, dem):
    """
    Adds an attribute 'fp_thick' containing an estimated floodplain thickness for each streams segment
    :param dn: string - path to drainage network shapefile. Should have 'width_low' and 'width_flood' attributes
    :param dem: string - path to dem
    :return:
    """

    network = gpd.read_file(dn)

    for i in network.index:
        print i
        if network.loc[i, 'fp_area'] != 0.:
            low_buf = network.loc[i, 'geometry'].buffer(network.loc[i, 'w_low']/2, cap_style=2)
            mid_buf = network.loc[i, 'geometry'].buffer(network.loc[i, 'w_bf']/2, cap_style=2)
            high_buf = network.loc[i, 'geometry'].buffer(network.loc[i, 'w_flood']/2, cap_style=2)
            fp_buf = high_buf.difference(mid_buf)

            low_zs = zonal_stats(low_buf, dem, stats='mean')
            fp_zs = zonal_stats(fp_buf, dem, stats='mean')

            chan_elev = low_zs[0].get('mean')
            fp_elev = fp_zs[0].get('mean')
            if chan_elev:
                if fp_elev:
                    network.loc[i, 'fp_thick'] = min(1., abs(fp_zs[0].get('mean') - low_zs[0].get('mean')))  # max of 1m, make parameter?
                else:
                    network.loc[i, 'fp_thick'] = 0.
            else:
                network.loc[i, 'fp_thick'] = 0.
        else:
            network.loc[i, 'fp_thick'] = 0.

    network.to_file(dn)

    return

wd = 'data/'
network = wd + 'Piru_network_500m.shp'
dem = wd + 'DEM_10m_Piru.tif'
est_fp_thickness(network, dem)

