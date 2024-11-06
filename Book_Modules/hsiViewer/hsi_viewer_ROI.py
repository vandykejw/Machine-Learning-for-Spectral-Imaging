import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib.colors as cm
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pyqtgraph as pg
import pandas as pd
from warnings import simplefilter
import numpy as np
import spectral
import time
import copy

# supress warnings from pandas when building DataFrames from ROI pixel spectra
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        
class viewer(QMainWindow):
    def __init__(self, im, stretch=[2,98]):
        # initiating GUI functions 
        window = pg.plot()      
        window.setWindowTitle('Initiating GUI') 
        window.close()         
        super().__init__()
        # set image data and metadata 
        self.stretch = stretch
        self.wl = np.asarray(im.bands.centers)
        self.imArr = im.Arr 
        self.nrows = im.Arr.shape[0]
        self.ncols = im.Arr.shape[1]
        self.nbands = im.Arr.shape[2] 
        self.nPix =  self.nrows*self.ncols
        self.imList = np.reshape(self.imArr, (self.nPix, self.nbands)) 
        # set basic image pixel coordinates values for ROIs (coordinates for all points in points_vstack)
        x, y = np.meshgrid(np.arange(self.ncols), np.arange(self.nrows))  # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        self.points_vstack = np.vstack((x, y)).T    
        # compute and set the geometry (shape and location) of the window
        self.computeGeometry()
        self.setGeometry(40, 40, int(self.w), int(self.h))  #loc_x, loc_y, width, height     
        # variables for ROIs
        self.ROImask_empty = np.zeros((self.nrows, self.ncols), dtype=bool)
        self.ROI_dict = {} # key = ROI ID Num, value = mask for the ROI
        self.ROI_polygons = []
        self.pcis = [] # list of the polygon points for the current polygon
        self.colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]
        
        # Create central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        # Create a horizontal layout area to hold the ROI creation buttons
        self.layout_top_menu = QHBoxLayout()     
        # button to start creating ROIs
        self.btn_ROIs = QPushButton("Collect ROIs")
        self.btn_ROIs.setCheckable(True)
        self.btn_ROIs.setMaximumWidth(100)
        self.btn_ROIs.clicked.connect(self.collectROIs)
        self.layout_top_menu.addWidget(self.btn_ROIs)
        # button to save ROIs
        self.btn_new_ROI = QPushButton("New ROI")
        self.btn_new_ROI.setMaximumWidth(100)
        self.btn_new_ROI.clicked.connect(self.newROI)
        self.layout_top_menu.addWidget(self.btn_new_ROI)
        # button to save ROIs
        self.btn_save_ROIs = QPushButton("Save ROIs")
        self.btn_save_ROIs.setMaximumWidth(100)
        self.btn_save_ROIs.clicked.connect(self.saveROIs)
        self.layout_top_menu.addWidget(self.btn_save_ROIs)
        # radio buttons for ROI selection method
        self.label_ROIselectionMethod = QLabel('  ROI selection method: ')  
        self.layout_top_menu.addWidget(self.label_ROIselectionMethod) 
        # select by polgons
        self.btn_roi_byPolygons = QRadioButton("Polygons")
        self.btn_roi_byPolygons.setChecked(True)
        self.layout_top_menu.addWidget(self.btn_roi_byPolygons)
        # select by points
        self.btn_roi_byPoints = QRadioButton("Points")
        self.layout_top_menu.addWidget(self.btn_roi_byPoints)
        # add a stretch to fill in the rest of the area in the layout
        self.layout_top_menu.addStretch(1)
        
        # list widget for selecting ROIs
        self.ROI_table = QTableWidget()
        self.ROI_table.setSelectionMode(QAbstractItemView.SingleSelection)
        nCols = 4
        nRows = 1
        self.ROI_table.setRowCount(nRows)
        self.ROI_table.setColumnCount(nCols)
        self.ROI_table.setHorizontalHeaderLabels(['Name', 'Color', '# Pixels','ROI Id num'])
        self.ROI_Id_num_count = 0
        self.ROI_table.hideColumn(3)
        # Set row contents
        # default start name
        self.ROI_table.setItem(0, 0, QTableWidgetItem("ROI "+str(0)))
        # start with red color
        item = QTableWidgetItem('  ')
        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        item.setBackground(QColor(250, 50, 50))
        self.current_color = QColor(250, 50, 50)
        self.ROI_table.setItem(0, 1, item)
        # start with 0 pixels
        item = QTableWidgetItem("0")
        item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
        self.ROI_table.setItem(0, 2, item)
        # start with unique Id num
        self.ROI_table.setItem(0, 3, QTableWidgetItem("ROI_num_"+str(self.ROI_Id_num_count)))
        self.ROI_dict["ROI_num_"+str(self.ROI_Id_num_count)] = self.ROImask_empty[:]
        index = self.ROI_table.model().index(0, 0)
        self.ROI_table.selectionModel().select(
            index, QItemSelectionModel.Select | QItemSelectionModel.Current)
        self.ROI_Id_num_count = self.ROI_Id_num_count + 1
        self.ROI_table.itemSelectionChanged.connect(self.ROI_table_selection_change)
        self.ROI_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.ROI_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.ROI_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.ROI_table.setStyleSheet("background-color: LightGrey;")
        self.ROI_table.setMaximumWidth(400)
        # create variable to hold polygons
        self.polygon_points_x = []
        self.polygon_points_y = []
        #self.polygonIm = QPolygon()
        self.polygonIm_points = []        
        
        # Center Area for image and ROI Table
        self.vbox_ROIs = QVBoxLayout()
        # create the frame object.
        self.box_ROIs_frame = QFrame()
        self.hbox_ROI_buttons = QHBoxLayout()
        self.hbox_ROI_buttons.addWidget(self.btn_new_ROI)
        self.hbox_ROI_buttons.addWidget(self.btn_save_ROIs)
        self.vbox_ROIs.addLayout(self.hbox_ROI_buttons)
        self.vbox_ROIs.addWidget(self.ROI_table)
        self.box_ROIs_frame.setLayout(self.vbox_ROIs)
        self.box_ROIs_frame.setMaximumWidth(300)
        self.box_ROIs_frame.hide()
        
        # create the display window with the image
        self.show_RGB() 
        
        # Create a vertical layout area with the buttons layout with image layout below
        self.layout_main_im = QVBoxLayout()
        self.layout_main_im.addLayout(self.layout_top_menu)
        self.layout_main_im.addWidget(self.imv)
        # Create a layout withe the buttons layout with image layout below
        self.layout_main = QHBoxLayout()
        self.layout_main.addLayout(self.layout_main_im)
        self.layout_main.addLayout(self.vbox_ROIs)
        self.layout_main.addWidget(self.box_ROIs_frame)
        self.central_widget.setLayout(self.layout_main)
        
        # Remove border around image
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.central_widget.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(0)
        
        # Set events
        self.imv.getImageItem().mouseClickEvent = self.click # create a spectral plot window if the image is clicked on
        
        self.show() 
        pg.exec()          
        
    def computeGeometry(self):
        aspect_ratio = self.nrows/self.ncols
        if aspect_ratio > 1:
            self.w = 1200
            self.h = self.w/aspect_ratio
        else:
            self.h = 1200
            self.w = aspect_ratio*self.h
            
    def show_RGB(self):              
        # create and show an RGB image in the viewer
         
        # determine the indices for the red, green, and blue bands
        self.index_red_band = np.argmin(np.abs(self.wl-650))
        self.index_green_band = np.argmin(np.abs(self.wl-550))
        self.index_blue_band = np.argmin(np.abs(self.wl-460))  
              
        # Create a numpy array for the RGB image with shape (nrows, ncols, 3)
        self.imRGB =np.zeros((self.nrows,self.ncols,3))
        self.imRGB[:,:,0] = self.stretch_arr(np.squeeze(self.imArr[:,:,self.index_red_band]))
        self.imRGB[:,:,1] = self.stretch_arr(np.squeeze(self.imArr[:,:,self.index_green_band]))
        self.imRGB[:,:,2] = self.stretch_arr( np.squeeze(self.imArr[:,:,self.index_blue_band]))
        self.imROI = copy.deepcopy(self.imRGB)
        self.imv = pg.image(self.imRGB)
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()
        #self.imv.setGeometry(100, 100, self.nrows, self.ncols) 
    
    def stretch_arr(self, arr):
        low_thresh_val = np.percentile(arr, self.stretch[0])
        high_thresh_val = np.percentile(arr, self.stretch[1])
        return np.clip(arr, a_min=low_thresh_val, a_max=high_thresh_val)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            row = self.ROI_table.currentRow()
            self.ROI_table.removeRow(row)
        else:
            super().keyPressEvent(event)
        
    def collectROIs(self):
        if self.btn_ROIs.isChecked():
            # show the RGB image in the viewer
            self.imv.setImage(self.imRGB, autoRange=False)
            self.box_ROIs_frame.show()
        else:
            # show the ROIs image in the viewer
            self.imv.setImage(self.imROI, autoRange=False)
            self.box_ROIs_frame.hide()
        
    def hex_to_rgb(self, value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    
    def newROI(self):
        rowPosition = self.ROI_table.rowCount()
        self.ROI_table.insertRow(rowPosition)
        # Set row contents
        # set deafult ROI name
        self.ROI_table.setItem(rowPosition, 0, QTableWidgetItem("ROI "+str(rowPosition)))
        # start with new color
        item = QTableWidgetItem('  ')
        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        rgb = self.hex_to_rgb(self.colors[rowPosition % 20])
        item.setBackground(QColor(rgb[0], rgb[1], rgb[2]))
        self.ROI_table.setItem(rowPosition, 1, item)
        # start with 0 pixels
        item = QTableWidgetItem("0")
        item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
        self.ROI_table.setItem(rowPosition, 2, item)
        # set the unique id num
        self.ROI_table.setItem(rowPosition, 3, QTableWidgetItem("ROI_num_"+str(self.ROI_Id_num_count)))
        self.ROI_dict["ROI_num_"+str(self.ROI_Id_num_count)] = self.ROImask_empty[:]
        self.ROI_Id_num_count = self.ROI_Id_num_count + 1
        # de-select all rows
        self.ROI_table.clearSelection()
        # select the newly added row
        index = self.ROI_table.model().index(rowPosition, 0)
        self.ROI_table.selectionModel().select(
            index, QItemSelectionModel.Select | QItemSelectionModel.Current)
        
    def saveROIs(self):
        # get output filename
        fname, extension = QFileDialog.getSaveFileName(self, "Choose output name", "C:\\spectral_data\\spectral_images", "PKL (*.pkl)")
        # return with no action if user selected "cancel" button
        if (len(fname)==0):
            return
        dataFrames = []
        for key in self.ROI_dict.keys():
            # if there is a row in the table associated with this key
            if len(self.ROI_table.findItems(key, Qt.MatchContains))>0:
                # get the spectra for this ROI
                spec = self.imList[np.reshape(self.ROI_dict[key], self.nPix),:]
                nPoints = spec.shape[0]
                # get the pixel locations for this ROI
                pixel_xy = self.points_vstack[np.reshape(self.ROI_dict[key], self.nPix),:]
                # get the row for this ROI
                item = self.ROI_table.findItems(key, Qt.MatchContains)[0]
                row = item.row()
                # get the color (red, green, blue, alpha) for this ROI
                color = self.ROI_table.item(row,1).background().color().name()
                # get the name for this ROI
                name = self.ROI_table.item(row, 0).text()
                df = pd.DataFrame()            
                df['Name'] = [name]*nPoints
                df['Color'] = [color]*nPoints
                df[['Pixel_x','Pixel_y']] = pixel_xy
                df[list(self.wl)] = spec  
                dataFrames.append(df) 
        df = pd.concat(dataFrames)
        df.to_pickle(fname)

    def ROI_table_selection_change(self):
        if len(self.ROI_table.selectedItems())==0:
            return
        # get the first selected item
        item = self.ROI_table.selectedItems()[0]
        row = item.row()
        column = item.column()
        # if this is the color column, initiate color picker
        if column == 1:
            current_color = item.background().color()
            new_color = QColorDialog.getColor(initial = current_color)
            if new_color.isValid():
                item = QTableWidgetItem('  ')
                rgb = [new_color.red(),new_color.green(),new_color.blue()]
                item.setBackground(QColor(rgb[0], rgb[1], rgb[2]))
                self.ROI_table.setItem(row, 1, item)
            self.ROI_table.clearSelection()
            self.current_ROI_Id = self.ROI_table.item(row, 3).text()
            mask = self.ROI_dict[self.current_ROI_Id]
            self.imROI[:,:,0] = self.imROI[:,:,0]*(mask==0) + mask*float(rgb[0])/255
            self.imROI[:,:,1] = self.imROI[:,:,1]*(mask==0) + mask*float(rgb[1])/255
            self.imROI[:,:,2] = self.imROI[:,:,2]*(mask==0) + mask*float(rgb[2])/255
            self.imv.setImage(self.imROI, autoRange=False)
    
    def plot_polygon_path(self):
        self.pcis.append(pg.PlotCurveItem(x=self.polygon_points_x, y=self.polygon_points_y))
        self.imv.addItem(self.pcis[-1]) 
    
    def remove_polygon(self):
        for pci in self.pcis: 
            self.imv.removeItem(pci)
        self.pcis = []
        self.polygon_points_x = []
        self.polygon_points_y = []
        self.polygonIm_points = [] 
        

    def click(self, event):           
        event.accept()  
        
        # if the select ROIs button is pressed:
        if self.btn_ROIs.isChecked():

            # check if a row is selected:
            if (len(self.ROI_table.selectedItems()) > 0):
                # get information for the row and the event
                item = self.ROI_table.selectedItems()[0] # get the selected item from the table
                row = item.row()# get the row for the selected item
                roi_color = self.ROI_table.item(row, 1).background().color()# get the color (from columne 1) for the selected item)
                self.current_ROI_Id = self.ROI_table.item(row, 3).text()# get the ROI id for the selected ROI row
                pos = event.pos() # get the position of the event
                x,y = int(pos.x()),int(pos.y()) # get the x,y pixel co0rdinates for the location  
                
                # check if the select-by-points radio button is checked
                if self.btn_roi_byPoints.isChecked():                    
                    self.remove_polygon()
                    
                    # if left button was clicked
                    if event.button() == Qt.LeftButton:
                        # add the new point to the ROI mask
                        self.ROI_dict[self.current_ROI_Id][x,y] = True
                        # color the new point in the ROI image                        
                        self.imROI[x,y,0] = float(roi_color.red())/255
                        self.imROI[x,y,1] = float(roi_color.green())/255
                        self.imROI[x,y,2] = float(roi_color.blue())/255
                        self.imv.setImage(self.imROI, autoRange=False)
                        # reset the number of points for the ROI in the table
                        item = QTableWidgetItem(str(np.sum(self.ROI_dict[self.current_ROI_Id])))
                        item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
                        self.ROI_table.setItem(row, 2, item)
                    
                    # right-button was clicked
                    else:  
                        # remove the new point from the ROI mask
                        self.ROI_dict[self.current_ROI_Id][x,y] = False
                        # color the point in the ROI iamge the original color, then
                        # change it to an ROI color if it is an ROI                       
                        self.imROI[x,y,0] = self.imRGB[x,y,0]
                        self.imROI[x,y,1] = self.imRGB[x,y,1]
                        self.imROI[x,y,2] = self.imRGB[x,y,2]
                        for r in range(self.ROI_table.model().rowCount()):
                            ID_num = self.ROI_table.item(r, 3).text()
                            if self.ROI_dict[ID_num][x,y]:
                                print(ID_num)
                                roi_color = self.ROI_table.item(r, 1).background().color()
                                self.imROI[x,y,0] = float(roi_color.red())/255
                                self.imROI[x,y,1] = float(roi_color.green())/255
                                self.imROI[x,y,2] = float(roi_color.blue())/255
                        # set the image viewer with the ROI image
                        self.imv.setImage(self.imROI, autoRange=False)
                        # reset the number of points for the ROI in the table
                        item = QTableWidgetItem(str(np.sum(self.ROI_dict[self.current_ROI_Id])))
                        item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
                        self.ROI_table.setItem(row, 2, item)
                    
                # the select-by-polygon radio button is checked
                else:
                    # if left button was clicked
                    if event.button() == Qt.LeftButton:
                        # ADDING A POLYGON
                        # add the new point to the polygon points
                        #self.imROI[x,y,0] = float(roi_color.red())/255
                        #self.imROI[x,y,1] = float(roi_color.green())/255
                        #self.imROI[x,y,2] = float(roi_color.blue())/255
                        #self.imv.setImage(self.imROI, autoRange=False)
                        self.polygon_points_x.append(x)
                        self.polygon_points_y.append(y)
                        self.ROI_table.item(row, 2).setText(str(len(self.polygon_points_x)))
                        # plot the path for the current polygon path
                        self.plot_polygon_path()
                        self.polygonIm_points.append([y,x])
                    
                    # right-button was clicked
                    else:        
                        if len(self.polygonIm_points) > 1:
                            # determine pixels inside this polygon
                            p = Path(self.polygonIm_points)  # make a polygon
                            # Determine the points (coordinates listed in vstack) inside this polygon.
                            # This is how you make a 2d mask from the grid: mask = grid.reshape(self.im.ncols, self.im.nrows)
                            # add these pixel locations to the set of all locations for this ROI
                            grid = p.contains_points(self.points_vstack)  
                            # paint the points inside the polygon with the given color for this ROI
                            mask = grid.reshape(self.nrows, self.ncols)
                            mask = np.asarray(mask)
                            self.ROI_dict[self.current_ROI_Id] = self.ROI_dict[self.current_ROI_Id] + mask
                            self.imROI[:,:,0] = self.imROI[:,:,0]*(mask==0) + mask*float(roi_color.red())/255
                            self.imROI[:,:,1] = self.imROI[:,:,1]*(mask==0) + mask*float(roi_color.green())/255
                            self.imROI[:,:,2] = self.imROI[:,:,2]*(mask==0) + mask*float(roi_color.blue())/255
                            self.remove_polygon()
                            self.imv.setImage(self.imROI, autoRange=False)
                            # reset the number of points for the ROI in the table
                            item = QTableWidgetItem(str(np.sum(self.ROI_dict[self.current_ROI_Id])))
                            item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
                            self.ROI_table.setItem(row, 2, item)
                            
            else:
                # User clicked on image with ROI selection checked
                # but without a row in the ROI table selected.
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                # setting message for Message Box
                msg.setText("A row in the ROI table must be selected to draw an ROI.")
                # setting Message box window title
                msg.setWindowTitle("Select row in ROI table.")
                # declaring buttons on Message Box
                msg.setStandardButtons(QMessageBox.Ok )
                retval = msg.exec_()
                    
        # if select ROIs button is not pressed, plot the spectrum
        else:
            # get the coordinates of the point that was clicked
            pos = event.pos() # get the position of the event
            x,y = int(pos.x()),int(pos.y()) # get the x,y pixel co0rdinates for the location 
            
            # Check if a spectral plot window exists
            try: 
                specPlot_exists = self.specPlot.isVisible() # True if the window has been created and is open. False if it was created and clsoed
            except:
                specPlot_exists = False # If no self.specPlot was created, the 'try' command will go into the 'except' case.
            
            if specPlot_exists:
                self.ci = (self.ci + 1) % len(self.colors) # iterate to the next color
                color = self.colors[self.ci] # select the color
                spec = self.imArr[x,y,:].flatten() # get the selected spectrum from the hyperspectral image
                self.specPlot.plot(self.wl, spec, pen=color, name=f'Pixel [{x},{y}]') # add the new spectrum to the plot
            else:
                self.specPlot = pg.plot()
                self.specPlot.addLegend()
                self.ci = 0 # initialize the color index
                color = self.colors[self.ci] # select the color
                spec = self.imArr[x,y,:].flatten()# get the selected spectrum from the hyperspectral image
                self.specPlot.plot(self.wl, spec, pen=color, name=f'Pixel [{x},{y}]') # create a plot window with the selected spectrum
                self.specPlot.showButtons()
                self.specPlot.showGrid(True, True)
                self.specPlot.setLabels(title='Pixel Spectra', bottom='Wavelength')
                # making the spectral plot window wider
                x = self.specPlot.geometry().x()
                y = self.specPlot.geometry().y()
                w = self.specPlot.geometry().width()
                h = self.specPlot.geometry().height()
                self.specPlot.setGeometry(int(0.5*x), y, 2*w, h)
                self.specPlot.addLegend()