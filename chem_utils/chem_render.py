import sys
import os
from PyQt5 import QtWidgets, QtCore
import pyvista as pv
from pyvistaqt import QtInteractor
from chem_utils.path import Path


class MoleculeViewer(QtWidgets.QMainWindow):
    def __init__(self, directory, parent=None):
        super(MoleculeViewer, self).__init__(parent)
        self.directory = directory
        self.setup_ui()

    def setup_ui(self):
        self.frame = QtWidgets.QFrame()
        hlayout = QtWidgets.QHBoxLayout()

        # List Widget to show file names
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.setMaximumWidth(200)
        self.file_list_widget.currentItemChanged.connect(self.file_selected)
        hlayout.addWidget(self.file_list_widget)

        # PyVista plotter
        self.plotter = QtInteractor(self.frame)
        hlayout.addWidget(self.plotter.interactor)

        # Set the layout and central widget
        self.frame.setLayout(hlayout)
        self.setCentralWidget(self.frame)
        self.populate_file_list()

        self.setWindowTitle('Molecule Viewer')
        self.resize(800, 600)
        self.show()

    def populate_file_list(self):
        # Populate the list widget with xyz files from the given directory
        for filename in os.listdir(self.directory):
            if filename.endswith('.xyz'):
                self.file_list_widget.addItem(filename)

    def file_selected(self, current, previous):
        if current:
            file_path = os.path.join(self.directory, current.text())
            self.render_molecule(file_path)

    def render_molecule(self, file_path):
        # Clear the plotter before rendering a new molecule
        self.plotter.clear()

        # This part is where you need to integrate your render function
        path = Path.load(file_path)
        if len(path) == 1:
            path = path[0]

        # Assuming the render method from your Path class takes care of the plotting
        path.render(plotter=self.plotter, show=False)

        # Enable cell picking
        self.plotter.enable_cell_picking(
            callback=self.picking_callback, show_message=True, font_size=10)

        # Update the plotter
        self.plotter.update()

    def picking_callback(self, picker):
        """Callback function for when a cell is picked."""
        cell_id = picker.cell_id
        if cell_id < 0:
            # No cell was picked
            return

        # Here you handle the picked cell information.
        # You could retrieve and display data about the picked atom or bond.
        print(f"Picked cell with ID: {cell_id}")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # Replace this path with the directory you want to load the .xyz files from
    directory_path = '/home/ivan/Science/hannes_nanomotors/nanomotors/fluorene/0/B3LYP/6_31G_d_p/'

    window = MoleculeViewer(directory_path)
    sys.exit(app.exec_())
