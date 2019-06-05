# imports
import floodplain_thickness

# directory
dir = '/data'

# Inputs - fill in
network = dir + '.shp'  # name and extension of drainage network shapefile
dem = dir + '.tif'  # name and extension of DEM

# run model - do not modify below here
floodplain_thickness.est_fp_thickness(network, dem)
