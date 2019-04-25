import geopandas as gpd
from shapely.geometry import LineString


def extract_floodplain_area(network, floodplain, lg_buf=1500, med_buf=500, sm_buf=50):
    """

    :param network:
    :param floodplain:
    :param lg_buf:
    :param med_buf:
    :param sm_buf:
    :return:
    """

    dn = gpd.read_file(network)
    fp = gpd.read_file(floodplain)

    fp_areas = []

    for i in dn.index:
        seg = dn.loc[i]
        da = seg['Drain_Area']  # check that network has attribute
        geom = seg['geometry']

        print i

        ept1 = (geom.boundary[0].xy[0][0], geom.boundary[0].xy[1][0])
        ept2 = (geom.boundary[1].xy[0][0], geom.boundary[1].xy[1][0])
        line = LineString([ept1, ept2])

        if da >= 250:
            buf = line.buffer(lg_buf, cap_style=2)
        elif da < 250 and da >= 25:
            buf = line.buffer(med_buf, cap_style=2)
        else:
            buf = line.buffer(sm_buf, cap_style=2)

        inters = buf.intersection(fp.loc[0, 'geometry'])  # make sure valley is single part
        fpa = inters.area - (seg['w_bf']*seg['Length_m'])  # check that network has attributes
        if fpa < 0:
            fpa = 0
        fp_areas.append(fpa)

    dn['fp_area'] = fp_areas

    dn.to_file(network)

    return