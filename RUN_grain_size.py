# imports
import class_D

# directory
dir = 'Piru/'

# Inputs - fill in
network = dir + 'Piru_network_1km.shp'  # name and extension of drainage network shapefile
grainsize = [dir + 'bluepoint_d_synth.csv']  # list of csv's containing grain size measurements
reachids = [56]  # list of reach IDs for associated grain size measurements above
width_table = dir + 'SC_width_table.csv'  # csv with channel width information; column headers: 'DA', 'Q', and 'w'

# run model
class_D.Dpred(network, grainsize, reachids, width_table)
