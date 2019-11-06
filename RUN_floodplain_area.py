# imports
from Scripts import floodplain_area

# directory
dir = 'data/'

# Inputs - fill in
network = dir + '.shp'  # name and extension of drainage network shapefile
floodplain = dir + '.shp'  # name and extension of floodplain/valley bottom shapefile
lg_buf = 2000  # maximum valley bottom width in high drainage area portions of network
med_buf = 500  # maximum valley bottom width in medium drainage area portions of network
sm_buf = 50  # maximum valley bottom width in small drainage area portions of the network

# run model - do not alter
floodplain_area.extract_floodplain_area(network, floodplain, lg_buf, med_buf, sm_buf)
