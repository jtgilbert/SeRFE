# imports
import confinement

# directory
dir = 'SP/'

# Inputs - fill in
network = dir + 'SP_network_500m.shp'
valley = dir + 'SP_VB.shp'

# run confinement model
inst = confinement.Confinement(network, valley, exag=0.08)
inst.confinement()
inst.update_width()
inst.update_area()
