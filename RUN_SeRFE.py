# imports
import geopandas as gpd
from Scripts import Visualizations, dynamic_model
import numpy as np

# directory
dir = 'data/'


# Inputs - fill these in
hydrograph = dir + '.csv'  # name and extension of csv file with hydrograph information filled in
width_table = dir + '.csv'  # .csv file with drainage area, discharge and width measurements
flow_exp = 0.84  # discharge-drainage area relationship exponent (can be found in plot produced from hydrology tool)
network = dir + '.shp'  # name and extension of drainage network shapefile
mannings_min = 0.03  # minimum Manning's n value for the basin (fine-grained reaches)
mannings_max = 0.05  # maximum Manning's n value for the basin (course-grained reaches)
bulk_density = 1.2  # average or estimated sediment bulk density in the basin
channel_storage = dir + 'channel_storage.npy'  # if you have a saved .npy file from a spinup run you can use it to set initial channel storage

outdf = dir + '.csv'  # name and extension for storing output dataframe
spinup = True  # True if running a spinup period in which floodplain height and slope values are updated without
                # saving any outputs. False if running model to store outputs


# Run Model - do not alter anything below this point

# check that network has all necessary attributes
stream_network = gpd.read_file(network)
attribs = ['Drain_Area', 'eff_DA', 'direct_DA', 'denude', 'confine', 'fp_area', 'fpt_mid', 'Q2 (cms)',
           'w_bf', 'Slope_mid', 'D_pred_mid', 'Length_m', 'g_shape', 'g_scale', 'dist_start', 'dist_end',
           'dist_g_sh', 'dist_g_sc']
for att in attribs:
    if att not in stream_network.columns:
        raise Exception('input stream network does not have all necessary attributes, run all sub-models first')

# maybe check for empty cells in hydrograph...

if spinup:
    serfe_run = dynamic_model.SerfeModel(hydrograph, width_table, flow_exp, network, mannings_min, mannings_max, bulk_density)
    chan_store = serfe_run.run_model(spinup=spinup)  # specify is running spin-up period or not
    # add code to save this file for us in future runs
    np.save(dir + 'channel_storage', chan_store)
    chan_store = dir + 'channel_storage.npy'
    serfe_run2 = dynamic_model.SerfeModel(hydrograph, width_table, flow_exp, network, mannings_min, mannings_max, bulk_density, chan_store=chan_store)
    output = serfe_run2.run_model(spinup=False)

    # save output dataframe to csv file
    if output is not None:
        output.to_csv(outdf)

    vis = Visualizations.Visualizations(outdf, network, hydrograph)
    vis.sum_plot('Qs_out_min')
    vis.sum_plot('Qs_out_mid')
    vis.sum_plot('Qs_out_max')

    Qs_norm_min = []
    Qs_norm_mid = []
    Qs_norm_max = []

    dn = gpd.read_file(network)
    for i in dn.index:
        qs_min = dn.loc[i, 'Qs_out_min'] / dn.loc[i, 'Drain_Area']
        Qs_norm_min.append(qs_min)
        qs_mid = dn.loc[i, 'Qs_out_mid'] / dn.loc[i, 'Drain_Area']
        Qs_norm_mid.append(qs_mid)
        qs_max = dn.loc[i, 'Qs_out_max'] / dn.loc[i, 'Drain_Area']
        Qs_norm_max.append(qs_max)

    dn['Qs_nor_min'] = Qs_norm_min
    dn['Qs_nor_mid'] = Qs_norm_mid
    dn['Qs_nor_max'] = Qs_norm_max
    dn.to_file(network)

else:
    serfe_run = dynamic_model.SerfeModel(hydrograph, width_table, flow_exp, network, mannings_min, mannings_max, bulk_density, chan_store=channel_storage)
    output = serfe_run.run_model(spinup=False)

    # save output dataframe to csv file
    if output is not None:
        output.to_csv(outdf)

    # update stream network with outputs
    vis = Visualizations.Visualizations(outdf, network, hydrograph)
    vis.sum_plot('Qs_out_min')
    vis.sum_plot('Qs_out_mid')
    vis.sum_plot('Qs_out_max')

    Qs_norm_min = []
    Qs_norm_mid = []
    Qs_norm_max = []

    dn = gpd.read_file(network)
    for i in dn.index:
        qs_min = dn.loc[i, 'Qs_out_min'] / dn.loc[i, 'Drain_Area']
        Qs_norm_min.append(qs_min)
        qs_mid = dn.loc[i, 'Qs_out_mid'] / dn.loc[i, 'Drain_Area']
        Qs_norm_mid.append(qs_mid)
        qs_max = dn.loc[i, 'Qs_out_max'] / dn.loc[i, 'Drain_Area']
        Qs_norm_max.append(qs_max)

    dn['Qs_nor_min'] = Qs_norm_min
    dn['Qs_nor_mid'] = Qs_norm_mid
    dn['Qs_nor_max'] = Qs_norm_max
    dn.to_file(network)
