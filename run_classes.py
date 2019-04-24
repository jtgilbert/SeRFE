# imports
import class_Q
import class_D
import confinement
import floodplain_thickness


network = ''  # shapefile
Q2_table = ''  # csv
grain_size = []  # list of csv's
f_sand = []  # list
reach_ids = []  # list
valley = ''  # shapefile
dem = ''  # raster

class_Q.HistoricQ(network, Q2_table)

class_D.Dpred(network, grain_size, f_sand, reach_ids)

conf_inst = confinement.Confinement(network, valley)
conf_inst.confinement()

floodplain_thickness.est_fp_thickness(network, dem)
