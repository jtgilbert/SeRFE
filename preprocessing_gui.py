from PyQt4 import QtGui
import sys
import design_preprocessing
from watershed_dem import watershed_dem
from drainage_area import drain_area
from network_topology import NetworkTopology
from network_attributes import add_da, add_slope
from channel_width import get_width_params, add_w

# from network_topology import network_topology


class TopologyTool(QtGui.QMainWindow, design_preprocessing.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        # tab 1 - project dem and clip to watershed boundary
        self.btnDEM_tab1.clicked.connect(lambda: self.file_browser(self.txtDEM_tab1))
        self.btnWats_tab1.clicked.connect(lambda: self.file_browser(self.txtWats_tab1))
        self.btnOutput_tab1.clicked.connect(lambda: self.file_save(self.txtOutput_tab1))
        self.btnCancel_tab1.clicked.connect(self.close)
        self.btnOk_tab1.clicked.connect(self.watsdem)

        # tab 2 - generate drainage area raster from dem
        self.btnDEM_tab2.clicked.connect(lambda: self.file_browser(self.txtDEM_tab2))
        self.btnOutput_tab2.clicked.connect(lambda: self.file_save(self.txtOutput_tab2))
        self.btnCancel_tab2.clicked.connect(self.close)
        self.btnOk_tab2.clicked.connect(self.drar)

        # tab 3 - generate network topology
        self.btnNetwork_tab3.clicked.connect(lambda: self.file_browser(self.txtNetwork_tab3))
        self.btnDEM_tab3.clicked.connect(lambda: self.file_browser(self.txtDEM_tab3))
        self.btnCancel_tab3.clicked.connect(self.close)
        self.btnOk_tab3.clicked.connect(self.topology)

        # tab 4 - add drainage area, slope and width attributes to network
        self.btnNetwork_tab4.clicked.connect(lambda: self.file_browser(self.txtNetwork_tab4))
        self.btnDEM_tab4.clicked.connect(lambda: self.file_browser(self.txtDEM_tab4))
        self.btnDA_tab4.clicked.connect(lambda: self.file_browser(self.txtDA_tab4))
        self.btnWidthTable_tab4.clicked.connect(lambda: self.file_browser(self.txtWidthTable_tab4))
        self.btnCancel_tab4.clicked.connect(self.close)
        self.btnOk_tab4.clicked.connect(self.attributes)

    def file_browser(self, txtControl):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '',
                                                     'Shapefiles (*.shp);; Rasters (*.img *.tif);; CSV (*.csv)')
        txtControl.setText(filename)

    def file_save(self, txtControl):
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Output File', '', 'All Files (*)')
        txtControl.setText(filename)

    def watsdem(self):
        dem = str(self.txtDEM_tab1.text())
        watershed = str(self.txtWats_tab1.text())
        epsg = int(self.txtEpsg_tab1.text())
        out = str(self.txtOutput_tab1.text())
        watershed_dem(dem, watershed, epsg, out)
        self.close()

    def drar(self):
        dem = str(self.txtDEM_tab2.text())
        drain_area_out = str(self.txtOutput_tab2.text())
        drain_area(dem, drain_area_out)
        self.close()

    def topology(self):
        network = str(self.txtNetwork_tab3.text())
        dem = str(self.txtDEM_tab3.text())
        NetworkTopology(network, dem)
        self.close()

    def attributes(self):
        network = str(self.txtNetwork_tab4.text())
        dem = str(self.txtDEM_tab4.text())
        drarea = str(self.txtDA_tab4.text())
        width_table = str(self.txtWidthTable_tab4.text())
        epsg = int(self.txtEpsg_tab4.text())
        add_da(network, drarea, epsg)
        add_slope(network, dem, epsg)
        a_low, b_low, a_bf, b_bf, a_flood, b_flood = get_width_params(width_table)
        add_w(network, a_low, b_low, a_bf, b_bf, a_flood, b_flood, epsg)
        self.close()


def main():
    app = QtGui.QApplication(sys.argv)
    form = TopologyTool()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()