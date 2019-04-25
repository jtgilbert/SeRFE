# imports
import confinement

# directory
dir = '/home/jordan/Documents/Geoscience/GIS/Piru_SeRFE_data/'

# Inputs - fill in
network = dir + 'Piru_network.shp'
valley = dir + 'Piru_valley.shp'

# run confinement model
inst = confinement.Confinement(network, valley, exag=0.1)
inst.confinement()
inst.update_width()
#inst.update_area()
