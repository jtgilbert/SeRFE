# imports
from Scripts import sed_delivery_params

# directory
dir = 'data/'

# Inputs - fill in
dem = dir + '.tif'  # name and extension of DEM
slope_out = dir + '.tif'  # if small DEM (e.g. HUC 12) set calc_slope = True and specify output name and extension,
                          # otherwise, use GIS to derive slope raster and specify name and extension of that raster.
network = dir + '.shp'  # name and extension of drainage network shapefile
neighborhood = 500  # neighborhood distance for calculating local gradient (m)
g_shape = 8  # gamma shape parameter (for erosion rates)
g_scale_min = 0.1  # minimum gamma scale parameter (for erosion rates)
g_scale_max = 0.15  # maximum gamma scale parameter (for erosion rates)
calc_slope = False  # if DEM is small (e.g. HUC 12) set True and specify output name and extension in slope_out param

# run model - do not modify below here
sed_delivery_params.SedDeliveryParams(dem, slope_out, network, neighborhood, g_shape, g_scale_min, g_scale_max, calc_slope)
