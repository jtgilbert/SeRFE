""" This function creates a drainage area raster from an input dem. """

# imports
import pygeoprocessing.routing as rt
import rasterio
import os


def drain_area(dem, drain_area_out):
    """
    Creates a raster where each pixel represents the contributing
    upstream drainage area in km2. DEM should be in a desired projected coordinate system.
    PARAMS
    :dem: string - path to dem raster file
    :drain_area_out: string - path to output drainage area raster file
    """

    # sets up intermediate file uris
    os.chdir(os.path.dirname(dem))
    filled = 'filled.tif'
    fd = 'fd.tif'
    fa = 'fa.tif'

    # geoprocessing functions
    rt.fill_pits((dem, 1), filled)
    rt.flow_dir_mfd((filled, 1), fd)
    rt.flow_accumulation_mfd((fd, 1), fa)

    # convert flow accumulation to drainage area
    flow_acc = rasterio.open(fa)
    resolution = flow_acc.res[0]
    flow_acc_array = flow_acc.read(1)
    dr_area_array = (flow_acc_array * resolution**2)/1000000.0

    # write output file
    profile = flow_acc.profile

    with rasterio.open(drain_area_out, 'w', **profile) as dst:
        dst.write(dr_area_array, 1)

    # delete intermediate files
    # os.remove(filled)
    # os.remove(fd)
    # os.remove(fa)

    return
