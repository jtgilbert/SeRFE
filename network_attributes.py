# imports
import geopandas as gpd
from rasterstats import zonal_stats
from shapely.geometry import Point
import network_topology as nt


def add_da(network, da, crs_epsg):
    """
    This function attributes each reach of a drainage network with a value representing the
    contributing upstream drainage area.
    PARAMS
    :network: string - path to a drainage network shapefile
    :da: string - path to a drainage area raster
    :crs_epsg: int - epsg number for output spatial reference
    :return:
    """

    # convert epsg number into crs dict
    sref = {'init': 'epsg:{}'.format(crs_epsg)}

    # read in network and check for projection
    flowlines = gpd.read_file(network)
    if flowlines['geometry'].crs == sref:
        pass
    else:
        flowlines = flowlines.to_crs(sref)

    # list to store da values
    da_list = []

    # iterate through each network segment, obtain da value and add to list
    for i in flowlines.index:
        # find and buffer segment midpoint to account for positional inaccuracy between da raster and network
        seg_geom = flowlines.loc[i, 'geometry']
        pos = int(len(seg_geom.coords.xy[0])/2)
        mid_pt_x = seg_geom.coords.xy[0][pos]
        mid_pt_y = seg_geom.coords.xy[1][pos]

        pt = Point(mid_pt_x, mid_pt_y)
        buf = pt.buffer(30)

        # get max drainage area value within buffered midpoint
        zs = zonal_stats(buf, da, stats='max')
        da_value = zs[0].get('max')

        da_list.append(da_value)

    # add da values to network attribute table
    flowlines['Drain_Area'] = da_list

    # check for segments with lower DA value than upstream segment; need to find more eloquent way to do this than
    # just repeating it three times in the code
    f = 1
    while f != 0:
        for i in flowlines.index:
            rid_us = flowlines.loc[i, 'rid_us']

            for j in flowlines.index:
                if flowlines.loc[j, 'rid'] == rid_us:
                    iden = flowlines.index[j]
                else:
                    iden = None

                if iden is not None:
                    if flowlines.loc[i, 'Drain_Area'] < flowlines.loc[iden, 'Drain_Area']:
                        flowlines.loc[i, 'Drain_Area'] = flowlines.loc[iden, 'Drain_Area'] + 0.1
                        f = 1
                else:
                    f = 0

    f = 1
    while f != 0:
        for i in flowlines.index:
            rid_us = flowlines.loc[i, 'rid_us']

            for j in flowlines.index:
                if flowlines.loc[j, 'rid'] == rid_us:
                    iden = flowlines.index[j]
                else:
                    iden = None

                if iden is not None:
                    if flowlines.loc[i, 'Drain_Area'] < flowlines.loc[iden, 'Drain_Area']:
                        flowlines.loc[i, 'Drain_Area'] = flowlines.loc[iden, 'Drain_Area'] + 0.1
                        f = 1
                else:
                    f = 0

    f = 1
    while f != 0:
        for i in flowlines.index:
            rid_us = flowlines.loc[i, 'rid_us']

            for j in flowlines.index:
                if flowlines.loc[j, 'rid'] == rid_us:
                    iden = flowlines.index[j]
                else:
                    iden = None

                if iden is not None:
                    if flowlines.loc[i, 'Drain_Area'] < flowlines.loc[iden, 'Drain_Area']:
                        flowlines.loc[i, 'Drain_Area'] = flowlines.loc[iden, 'Drain_Area'] + 0.1
                        f = 1
                else:
                    f = 0

    flowlines.to_file(network)

    return


def add_slope(network, dem, crs_epsg):
    """
    Extracts elevation from the start and end of each network segment to
    calculate slope and add it to network attributes.
    PARAMS
    :network: string - path to drainage network shapefile
    :dem: string - path to dem raster file
    :crs_epsg: int - epsg number for output spatial reference
    """

    # convert epsg number into crs dict
    sref = {'init': 'epsg:{}'.format(crs_epsg)}

    # read in network and check for projection
    flowlines = gpd.read_file(network)
    if flowlines['geometry'].crs == sref:
        pass
    else:
        flowlines = flowlines.to_crs(sref)

    # create a list to store slope values
    slope = []

    # iterate through each network segment, calculate slope and add to list
    for i in flowlines.index:
        # obtain the coordinates of the end points of each line segment
        seg_geom = flowlines.loc[i, 'geometry']
        length = seg_geom.length

        x_coord1 = seg_geom.boundary[0].xy[0][0]
        y_coord1 = seg_geom.boundary[0].xy[1][0]
        x_coord2 = seg_geom.boundary[1].xy[0][0]
        y_coord2 = seg_geom.boundary[1].xy[1][0]

        # create points at the line end points
        pt1 = Point(x_coord1, y_coord1)
        pt2 = Point(x_coord2, y_coord2)

        # buffer the points to account for positional innacuracy between DEM and network
        buf1 = pt1.buffer(20)
        buf2 = pt2.buffer(20)

        # obtain elevation values within the buffers
        zs1 = zonal_stats(buf1, dem, stats='min')
        zs2 = zonal_stats(buf2, dem, stats='min')
        elev1 = zs1[0].get('min')
        elev2 = zs2[0].get('min')

        # calculate the slope of each reach and append it to the list
        slope_value = abs(elev1-elev2)/length
        if slope_value < 0.001:
            slope_value = 0.001

        slope.append(slope_value)

    # add slope values to network attribute table
    flowlines['Slope'] = slope
    flowlines.to_file(network)

    return
