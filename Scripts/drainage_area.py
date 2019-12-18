""" This function creates a drainage area raster from an input dem. """

# imports
import richdem as rd


def drain_area(dem, drain_area_out):
    """
    Creates a raster where each pixel represents the contributing
    upstream drainage area in km2. DEM should be in a desired projected coordinate system.
    PARAMS
    :dem: string - path to dem raster file
    :drain_area_out: string - path to output drainage area raster file
    """

    dem_in = rd.LoadGDAL(dem)
    rd.FillDepressions(dem_in, epsilon=True, in_place=True)
    accum_d8 = rd.FlowAccumulation(dem_in, method='D8')
    da = accum_d8 * (accum_d8.geotransform[1] ** 2 / 1000000)
    rd.SaveGDAL(drain_area_out, da)

    return
