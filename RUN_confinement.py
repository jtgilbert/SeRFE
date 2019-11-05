# imports
from Scripts import confinement

# directory
dir = 'Piru/'

# Inputs - fill in
network = dir + 'Piru_network_1km.shp'  # name and extension of drainage network shapefile
valley = dir + 'Piru_VB2.shp'  # name and extension of floodplain/valley bottom shapefile

# run confinement model - do not modify anything below
inst = confinement.Confinement(network, valley, exag=0.01)
inst.confinement()
inst.update_area()
