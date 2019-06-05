# imports
import confinement

# directory
dir = 'data/'

# Inputs - fill in
network = dir + '.shp'  # name and extension of drainage network shapefile
valley = dir + '.shp'  # name and extension of floodplain/valley bottom shapefile

# run confinement model - do not modify anything below
inst = confinement.Confinement(network, valley, exag=0.08)
inst.confinement()
inst.update_width()
inst.update_area()
