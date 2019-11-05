from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
import sys
from Scripts import preprocessing_d
from Scripts.watershed_dem import watershed_dem
from Scripts.drainage_area import drain_area
from Scripts.network_topology import NetworkTopology
from Scripts.network_attributes import add_da, add_slope

# from network_topology import network_topology


class TopologyTool(QMainWindow, preprocessing_d.Ui_MainWindow):
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
        self.btnCancel_tab4.clicked.connect(self.close)
        self.btnOk_tab4.clicked.connect(self.attributes)

    def file_browser(self, txtControl):
        filename = QFileDialog.getOpenFileName(self, 'Open File', '',
                                                     'Shapefiles (*.shp);; Rasters (*.img *.tif);; CSV (*.csv)')
        txtControl.setText(filename[0])

    def file_save(self, txtControl):
        filename = QFileDialog.getSaveFileName(self, 'Output File', '', 'All Files (*)')
        txtControl.setText(filename[0])

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
        epsg = int(self.txtEpsg_tab4.text())
        add_da(network, drarea, epsg)
        add_slope(network, dem, epsg)
        self.close()


def main():
    app = QApplication(sys.argv)
    form = TopologyTool()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()