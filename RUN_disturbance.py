# imports
from Scripts import disturbance

# directory
dir = 'data/'

# stream network - fill in name and extension of drainage network shapefile
network = dir + '.shp'

# ADD DISTURBANCES

# 1. SEDIMENT DECREASING DISTURBANCE (e.g. dam)
# Inputs - fill in
segid1 = []  # List of segment IDs for each segment at location of sediment reduction (obtain from GIS)
new_da = True  # leave true to update effective drainage area below dams

# 2. SEDIMENT INCREASING DISTURBANCE (e.g. wildfire)
# Inputs - fill in
segid2 = []  # list of segment IDs for each segment with increased sedimentation from disturbance (obtain from GIS)
dist_start = []  # time step to begin disturbance. This can be a single value if it is the same for all disturbances,
                  # or a list of the same length as segment IDs, where each value is the start time step for that
                  # specific segment.
dist_end = []  # time step to end disturbance. This can be a single value if it is the same for all disturbances,
                # or a list of the same length as segment IDs, where each value is the end time step for that specific
                # segment.
new_denude = [[]] # gamma shape and scale values for new denudation/erosion rates from disturbance. Can be single
                      # values for all disturbance (i.e. [shape, scale]) or a list of values for each
                      # segment in the segment ID list (i.e. [shape, scale],[shape, scale]... for each segment).

#  run disturbance - do not modify
dist = disturbance.Disturbances(network)
if len(segid1) > 0:
    dist.add_disturbance(segid1, new_da=new_da)

if len(segid2) > 0:
    if len(dist_start) > 1:
        if len(segid2) != len(dist_start):
            raise Exception('segment ID list and start time list have different lengths')
        if len(dist_start) != len(dist_end):
            raise Exception('disturbance start time list and end time list have different lengths')
    if len(new_denude) > 1:
        if len(segid2) != len(new_denude):
            raise Exception('segment ID list and new denude list have different lengths')

    dist.add_disturbance(segid2, dist_start=dist_start, dist_end=dist_end, new_denude=new_denude)

dist2 = disturbance.Disturbances(network)
dist2.update_direct_da()

