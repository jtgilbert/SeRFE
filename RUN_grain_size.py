# imports
from Scripts import class_D2

# directory
dir = 'data/'

# Inputs - fill in
network = dir + '.shp'  # name and extension of drainage network shapefile
grainsize = []  # list of csv's containing grain size measurements
reachids = []  # list of reach IDs for associated grain size measurements above
width_table = dir + '.csv'  # csv with channel width information; column headers: 'DA', 'Q', and 'w'

# run model
class_D2.Dpred(network, grainsize, reachids, width_table)
