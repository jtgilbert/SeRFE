# imports
from Scripts import floodplain_thickness

# directory
dir = 'SC/'

# Inputs - fill in
network = dir + 'SC_serfe_network.shp'  # name and extension of drainage network shapefile
valley = dir + 'SC_VB.shp'  # name and extension of valley bottom shapefile
dem = dir + 'DEM_10m_SantaClara.tif'  # name and extension of DEM
min_thickness = 1.5  # a minimum floodplain thickness value (m)
max_thickness = 3  # a maximum floodplain thickness value (m)

# run model - do not modify below here
floodplain_thickness.est_fp_thickness(network, valley, dem, min_thickness, max_thickness)
