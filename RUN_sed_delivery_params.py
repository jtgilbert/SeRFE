# imports
import sed_delivery_params

# directory
dir = 'Piru/'

# Inputs - fill in
dem = dir + 'DEM_10m_Piru.tif'  # name and extension of DEM
slope_out = dir + 'slope.tif'  # if small DEM (e.g. HUC 12) set calc_slope = True and specify output name and extension,
                          # otherwise, use GIS to derive slope raster and specify name and extension of that raster.
network = dir + 'Piru_network_1km.shp'  # name and extension of drainage network shapefile
neighborhood = 500  # neighborhood distance for calculating local gradient (m)
g_min = 3  # minimum gamma shape parameter (for erosion rates)
g_max = 5.5  # maximum gamma shape parameter (for erosion rates)
g_scale = 0.3  # gamma scale parameter (for erosion rates)
calc_slope = False  # if DEM is small (e.g. HUC 12) set True and specify output name and extension in slope_out param

# run model - do not modify below here
sed_delivery_params.SedDeliveryParams(dem, slope_out, network, neighborhood, g_min, g_max, g_scale, calc_slope)
