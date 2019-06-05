# imports
import class_D

# directory
dir = 'data/'

# Inputs - fill in
network = dir + '.shp'  # name and extension of drainage network shapefile
grainsize = [dir + '.csv']  # list of csv's containing grain size measurements
f_sand = []  # list of values for fraction of bed that is sand associated with each grain size measurement above
reachids = []  # list of reach IDs for associated grain size measurements above
f_sand_default = 0.25  # a default value for the fraction of sand in the bed

# run model
class_D.Dpred(network, grainsize, f_sand, reachids, f_sand_default)
