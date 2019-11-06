# imports
from Scripts import Visualizations

# directory
dir = '/data'

# Inputs - fill in
data_frame = dir + '.csv'  # name and extension of model output dataframe
network = dir + '.shp'  # name and extension of drainage network shapefile
hydrographs = dir + '.csv'  # name and extension of hydrographs used in model
segment_id = 0  # the segment ID of the segment for which you wish to produce plots
discharge = False  # boolean - produce a discharge time series plot for the segment
storage = False  # boolean - produce a sediment storage time series plot for the segment
csr = False  # boolean - produce a capacity to supply ratio time series plot for the segment

# produce desired plots
vis = Visualizations.Visualizations(data_frame, network, hydrographs)

if discharge:
    vis.plot_time_series(segment_id, 'Q')

if storage:
    vis.plot_storage(segment_id)

if csr:
    vis.plot_csr_time_series(segment_id)
