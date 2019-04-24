# imports
import confinement

# directory
dir = 'data/'

# Inputs - fill in
network = dir + 'SC_network.shp'
valley = dir + 'SC_VB.shp'

# run confinement model
inst = confinement.Confinement(network, valley, exag=0.08)
inst.confinement()
inst.update_width()
#inst.update_area()
