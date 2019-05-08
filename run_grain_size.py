# imports
import class_D

# directory
dir = 'SP/'

# Inputs - fill in
network = dir + 'SP_network_500m.shp'
grainsize = [dir + 'SantaPaula_ThomasAquinas_D.csv']
f_sand = [0.1]
reachids = [16]
f_sand_default = 0.25

# run model
class_D.Dpred(network, grainsize, f_sand, reachids, f_sand_default)
