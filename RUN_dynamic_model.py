# imports
import dynamic_model
import geopandas as gpd
import Visualizations

# directory
dir = '/data'


# Inputs - fill these in
hydrograph = dir + '.csv'  # name and extension of csv file with hydrograph information filled in
flow_exp = 0.84  # discharge-drainage area relationship exponent (can be found in plot produced from hydrology tool)
network = dir + '.shp'  # name and extension of drainage network shapefile
mannings_n = 0.4  # average Manning's n value for the basin
tl_factor = 16.  # the total load factor to convert bedload to total load transport capacity
outdf = dir + '.csv'  # name and extension for storing output dataframe
spinup = False  # True if running a spinup period in which floodplain height and slope values are updated without
                # saving any outputs. False if running model to store outputs

# Run Model - do not alter anything below this point

# check that network has all necessary attributes
stream_network = gpd.read_file(network)
attribs = ['Drain_Area', 'eff_DA', 'direct_DA', 'denude', 'confine', 'fp_area', 'fp_thick', 'f_sand', 'Q2 (cms)',
           'w_low', 'w_bf', 'w_flood', 'Slope', 'D_pred', 'Length_m', 'g_shape', 'g_scale', 'dist_start', 'dist_end',
           'dist_g_sh', 'dist_g_sc']
for att in attribs:
    if att not in stream_network.columns:
        raise Exception('input stream network does not have all necessary attributes, run all sub-models first')

# maybe check for empty cells in hydrograph...

serfe_run = dynamic_model.SerfeModel(hydrograph, flow_exp, network, mannings_n, tl_factor)
output = serfe_run.run_model(spinup=spinup)  # specify is running spin-up period or not

# save output dataframe to csv file
if output is not None:
    output.to_csv(outdf)

# update stream network with outputs
vis = Visualizations.Visualizations(outdf, network, hydrograph)
vis.sum_plot('Qs_out')
vis.delta_storage_plot()
vis.csr_integrate()

Qs_norm = []
dn = gpd.read_file(network)
for i in dn.index:
    qs = dn.loc[i, 'Qs_out'] / dn.loc[i, 'Drain_Area']
    Qs_norm.append(qs)

dn['Qs_norm'] = Qs_norm
dn.to_file(network)
