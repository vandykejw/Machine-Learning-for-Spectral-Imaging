import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import numpy as np
import pyqtgraph as pg
import spectral
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)


class viewer(QMainWindow):
    def __init__(self, im, layers={}): 
        # initiating GUI functions 
        window = pg.plot()      
        window.setWindowTitle('Initiating GUI') 
        window.close()         
        super().__init__()
        # initiate variables
        self.wl = np.asarray(im.bands.centers)
        self.imArr = im.Arr  
        self.layers = layers
        self.nrows = im.Arr.shape[0]
        self.ncols = im.Arr.shape[1]
        self.nbands = im.Arr.shape[2]    
        # compute and set the geometry (shape and location) of the window
        self.computeGeometry()
        self.setGeometry(40, 40, int(self.w), int(self.h))  #loc_x, loc_y, width, height  
        # create the display window with the image
        self.show_RGB()  
        # set plot information
        self.colors = ['r', 'b','g','c','m','y', 'w'] # colors to iterate through for plotting spectra - (can be modified with any colors - see https://pyqtgraph.readthedocs.io/en/latest/_modules/pyqtgraph/functions.html#mkColor for options)

        # Create central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        # Create buttons from the inpout layers
        self.setLayerButtons()
        # Add stretch and band-scrolling button
        self.layout_top_menu.addStretch(1)
        self.layout_top_menu.addWidget(self.buttonViewBands)
        # Create a layout withe the buttons layout with image layout below
        self.layout_main = QVBoxLayout()
        self.layout_main.addLayout(self.layout_top_menu)
        self.layout_main.addWidget(self.imv)
        self.central_widget.setLayout(self.layout_main)
        
        # Remove border around image
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.central_widget.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(0)
         
        # Set events
        self.imv.getImageItem().mouseClickEvent = self.click # create a spectral plot window if the image is clicked on
        
        # Show the Pyt widgets and execute the qtgraph widgets
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

    def setLayerButtons(self):        
        self.buttonViewBands = QPushButton('Scroll Bands', self)
        self.buttonViewBands.setMaximumWidth(100)
        self.buttonViewBands.clicked.connect(self.onViewBandsClick)             
        # Create a layout area at the top of the central_widget for the buttons
        self.layout_top_menu = QHBoxLayout()     
        button = QPushButton('RGB Image', self)
        button.setMaximumWidth(100)
        button.clicked.connect(self.onButtonAddRGBimClick)
        self.layout_top_menu.addWidget(button)
        # add a button to add each layer in the dictionary 
        for layer_key in self.layers.keys():
            button = QPushButton(layer_key, self)
            button.setMaximumWidth(100)
            button.clicked.connect(self.onButtonAddLayerClick)  
            self.layout_top_menu.addWidget(button)
            
    def onViewBandsClick(self):
        self.scrollBandArr = np.swapaxes(np.swapaxes(self.imArr, 0, 2), 1, 2)
        self.imv.setImage(self.scrollBandArr, autoRange=True)
        
    def onButtonAddRGBimClick(self):
        self.imv.setImage(self.imRGB, autoRange=False)

    def onButtonAddLayerClick(self):
        sending_button = self.sender()
        layer_name = str(sending_button.text())
        if isinstance(self.layers[layer_name], list):
            if len(self.layers[layer_name])==1:
                layer_im = np.squeeze(self.layers[layer_name])
                cmap = 'jet'
            else:
                layer_im = np.squeeze(self.layers[layer_name][0])
                cmap = self.layers[layer_name][1]
        else:
            layer_im = np.squeeze(self.layers[layer_name])
            cmap = 'jet'
        if len(layer_im.shape) == 2:
            # to create RGB values for a class image
            #layer_im = self.create_layer_RGB_image(layer_im, cmap)
            self.imv.setImage(layer_im, autoRange=False)
        self.imv.setImage(layer_im, autoRange=False)
        
    def create_layer_RGB_image(self, layer_im, cmap):
        nRows,nCols = layer_im.shape
        layer_im_rgb = np.zeros((nRows,nCols,4))
        colorMaker = MplColorHelper(cmap, np.min(layer_im), np.max(layer_im))
        for val in np.unique(layer_im):
            layer_im_rgb[layer_im==val] = colorMaker.get_rgb(val)
        return layer_im_rgb        
        
    def show_RGB(self):       
        # create and show an RGB image in the viewer         
        # determine the indices for the red, green, and blue bands
        self.index_red_band = np.argmin(np.abs(self.wl-650))
        self.index_green_band = np.argmin(np.abs(self.wl-550))
        self.index_blue_band = np.argmin(np.abs(self.wl-460))                
        # Create a numpy array for the RGB image with shape (nrows, ncols, 3)
        self.imRGB =np.zeros((self.nrows,self.ncols,3))
        self.imRGB[:,:,0] = np.squeeze(self.imArr[:,:,self.index_red_band])
        self.imRGB[:,:,1] = np.squeeze(self.imArr[:,:,self.index_green_band])
        self.imRGB[:,:,2] = np.squeeze(self.imArr[:,:,self.index_blue_band])
        # Set the numpy RGB image in the viewer
        self.imv = pg.image(self.imRGB, autoRange=True)

    def click(self, event):    
        # plot the spectrum for the clicked pixel location
        
        event.accept()  
        pos = event.pos() # get the position of the event
        x,y = int(pos.x()),int(pos.y()) # get the x,y pixel corrdinates for the location
        print (f'x,y = [{x},{y}]')
        
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
            self.specPlot.setGeometry(int(0.5*x), 2*y, 2*w, h)
            self.specPlot.addLegend()
            
            