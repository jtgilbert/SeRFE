# imports
from Scripts import class_D

# directory
dir = 'SC/'

# Inputs - fill in
network = dir + 'SC_network_1km.shp'  # name and extension of drainage network shapefile
grainsize = [dir + 'sc_countyline_d.csv', dir + 'sespe_d.csv', dir + 'bluepoint_d_synth.csv', dir + 'sp_d4.csv']  # list of csv's containing grain size measurements
reachids = [312, 522, 872, 414]  # list of reach IDs for associated grain size measurements above
width_table = dir + 'SC_width_table.csv'  # csv with channel width information; column headers: 'DA', 'Q', and 'w'

# run model
class_D.Dpred(network, grainsize, reachids, width_table)
