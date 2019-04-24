""""""

# imports
import class_Q
import class_Qs
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


# directory
dir = 'data/'

# Inputs - fill these in
network = dir + ''
a = None
b = None
updated_Q2_table = dir + 'updated_Q2.csv'
qs_table = dir + 'qs.csv'
updated_qs_table = dir + 'updated_qs.csv'

# Run Model - do not alter anything below this point
q_inst = class_Q.UpdatedQ(network, updated_Q2_table, a, b)
q_inst.update_above_dist()
q_inst.update_affected_reaches()
q_inst.update_below_confluences()

class_Qs.BaselineQs(network, qs_table)

qs_inst = class_Qs.UpdatedQs(network, updated_qs_table)
qs_inst.unaffected_reaches()
qs_inst.ds_reductions()
qs_inst.update_below_confluences()

dn = gpd.read_file(network)

for i in dn.index:
    dn.loc[i, 'S*'] = np.sqrt((dn.loc[i, 'newQs']/dn.loc[i, 'Qs (t/yr)'])) * (dn.loc[i, 'Q2 (cms)']/dn.loc[i, 'newQ2'])

fig, ax = plt.subplots(figsize=(10, 10))
dn.plot(column='S*', ax=ax, cmap='RdYlBu')
plt.show()

dn.to_file(network)
