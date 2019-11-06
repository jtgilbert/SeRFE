# imports
import geopandas as gpd
from rasterstats import zonal_stats


def est_fp_thickness(dn, valley, dem, min_thickness=0.2, max_thickness=1.5):
    """
    Adds 3 attributes 'fpt_(min, mid, max)' to drainage network representing estimated floodplain thickness
    associated with each network segment based on a dem.
    :param dn: string - path to drainange network shapefile.
    :param valley: string - path to floodplain/valley bottom shapefile.
    :param dem: stinrg - path to DEM.
    :param min_thickness: a minimum floodplain thickness for segments with floodplains.
    :param max_thickness: a maximum floodplain thickness for segments with floodplains.
    :return:
    """

    network = gpd.read_file(dn)
    vb = gpd.read_file(valley)

    for i in network.index:
        print 'segment ', i, ' of ', len(network)
        if network.loc[i, 'confine'] < 1:
            chan_buf = network.loc[i, 'geometry'].buffer(network.loc[i, 'w_bf']/2, cap_style=2)
            lg_buf = network.loc[i, 'geometry'].buffer(network.loc[i, 'w_bf']*1.5, cap_style=2)
            vb_buf = vb.intersection(lg_buf)

            fp_buf = vb_buf.difference(chan_buf)

            low_zs = zonal_stats(chan_buf, dem, stats='mean')
            fp_zs = zonal_stats(fp_buf, dem, stats='mean')

            chan_elev = low_zs[0].get('mean')
            fp_elev = fp_zs[0].get('mean')
            if chan_elev:
                if fp_elev:
                    fp_thick = max(min_thickness, abs(fp_zs[0].get('mean') - low_zs[0].get('mean')))  # min of 0.2 m
                    network.loc[i, 'fpt_min'] = min(max_thickness, fp_thick)  # max of 1.5 m, make parameter?
                    network.loc[i, 'fpt_mid'] = min(max_thickness, fp_thick)
                    network.loc[i, 'fpt_max'] = min(max_thickness, fp_thick)
                else:
                    network.loc[i, 'fpt_min'] = 0.
                    network.loc[i, 'fpt_mid'] = 0.
                    network.loc[i, 'fpt_max'] = 0.
            else:
                network.loc[i, 'fpt_min'] = 0.
                network.loc[i, 'fpt_mid'] = 0.
                network.loc[i, 'fpt_max'] = 0.
        else:
            network.loc[i, 'fpt_min'] = 0.
            network.loc[i, 'fpt_mid'] = 0.
            network.loc[i, 'fpt_max'] = 0.

    network.to_file(dn)

    return
