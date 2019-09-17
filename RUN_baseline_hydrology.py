# imports
import class_Q

# directory where data is stored,
dir = 'SC/'

# Inputs - fill in with the name and extension of input drainage network
network = dir + 'SC_network_1km.shp'  # shapefile
Q2_table = dir + 'reference_Q2.csv'  # csv

# Run Model - do not alter anything below this point
class_Q.HistoricQ(network, Q2_table)