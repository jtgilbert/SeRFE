# imports
import class_D

# directory
dir = 'data/'

# Inputs - fill in
network = dir + 'SC_network.shp'
grainsize = [dir + 'Piru_Hardluck_D.csv', dir + 'Piru_Osito_D.csv', dir + 'Piru_BluePoint_D.csv', dir + 'SantaPaula_ThomasAquinas_D.csv']
f_sand = [0.25, 0.3, 0.35, 0.1]
reachids = [1511, 1192, 846, 397]
f_sand_default = 0.5

# run model
class_D.Dpred(network, grainsize, f_sand, reachids, f_sand_default)
