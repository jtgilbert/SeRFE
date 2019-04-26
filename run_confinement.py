# imports
import confinement

# directory
dir = 'data/'

# Inputs - fill in
network = dir + 'Mutau_network.shp'
valley = dir + 'Mutau_vb.shp'

# run confinement model
inst = confinement.Confinement(network, valley, exag=0.1)
inst.confinement()
inst.update_width()
inst.update_area()
