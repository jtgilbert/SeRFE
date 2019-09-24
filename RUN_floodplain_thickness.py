# imports
import floodplain_thickness

# directory
dir = 'Piru/'

# Inputs - fill in
network = dir + 'Piru_network_1km.shp'  # name and extension of drainage network shapefile
valley = dir + 'Piru_VB.shp'  # name and extension of valley bottom shapefile
dem = dir + 'DEM_10m_Piru.tif'  # name and extension of DEM
min_thickness = 0.3  # a minimum floodplain thickness value (m)
max_thickness = 2  # a maximum floodplain thickness value (m)

# run model - do not modify below here
floodplain_thickness.est_fp_thickness(network, valley, dem, min_thickness, max_thickness)
