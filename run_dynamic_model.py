"""Before running the dynamic model, the user should have completed at least the preprocessing steps and run the
D and Q classes to store all of the necessary information on the stream network. Additionally, the network should
be attributed with confinement values, floodplain area and thickness before running the model. """

# imports
import dynamic_model
import geopandas as gpd


# Inputs - fill these in
ws = 'data/'
hydrograph = ws + 'Piru_hydrographs.csv'
flow_exp = 0.84
network = ws + 'Piru_network_500m.shp'
mannings_n = 0.4
tl_factor = 16.
outdf = ws + 'piru_output_n4tl16.csv'

# Run Model - do not alter anything below this point

# check that network has all necessary attributes
stream_network = gpd.read_file(network)
attribs = ['Drain_Area', 'eff_DA', 'direct_DA', 'denude', 'confine', 'fp_area', 'fp_thick', 'f_sand', 'Q2 (cms)',
           'w_low', 'w_bf', 'w_flood', 'Slope', 'D_pred', 'Length_m', 'g_shape', 'g_scale']  # also need to make sure that anywhere confinement = 0 fp_area also = 0
for att in attribs:
    if att not in stream_network.columns:
        raise Exception('input stream network does not have all necessary attributes')

# maybe check for empty cells in hydrograph...

seba_run = dynamic_model.SebaModel(hydrograph, flow_exp, network, mannings_n, tl_factor)
output = seba_run.run_model()

output.to_csv(outdf)
