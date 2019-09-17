# imports
import class_D

# directory
dir = 'SP/'

# Inputs - fill in
network = dir + 'SP_network_500m.shp'  # name and extension of drainage network shapefile
grainsize = [dir + 'sp_d2.csv']  # list of csv's containing grain size measurements
reachids = [16]  # list of reach IDs for associated grain size measurements above
width_table = dir + 'SC_width_table.csv'  # csv with channel width information; column headers: 'DA', 'Q', and 'w'

# run model
class_D.Dpred(network, grainsize, reachids, width_table)
