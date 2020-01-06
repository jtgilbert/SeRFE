# imports
from Scripts import Visualizations

# directory
dir = 'data/'

# Inputs - fill in
output_table = dir + '.csv'  # output table from SeRFE dynamic model
network = dir + '.shp'  # drainage network shapefile
hydrograph = dir + '.csv'

# Fill in if you want CSR attributes/figures
CSR = False  # enter True if calculating CSR
CSR_start = None  # enter a time step to start CSR integration
CSR_end = None  # enter a time step to end CSR integration

# Fill in if you want to produce plots of storage through time for a given segment
storage_plot = False
storage_segment = None

# Fill in if you want storage change attributes/figures
store = False  # enter True if calculating storage change
store_start = None
store_end = None

# Fill in for to add an attribute for a certain time step
date_fig = False
time_step = None
attribute = None

# Fill in to produce a time series of CSR for a chosen segment
CSR_series = False
CSR_segment = None

# Do not edit
inst = Visualizations.Visualizations(output_table, network, hydrograph)

if CSR:
    inst.csr_integrate(CSR_start, CSR_end)

if storage_plot:
    inst.plot_storage(storage_segment)

if store:
    inst.d_storage_atts(store_start, store_end)
    inst.delta_storage_plot(store_start, store_end)

if date_fig:
    inst.date_fig(time_step, attribute)

if CSR_series:
    inst.plot_csr_time_series(CSR_segment)