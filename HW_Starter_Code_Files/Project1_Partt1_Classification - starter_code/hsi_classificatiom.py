import matplotlib.pyplot as plt
import scripts_M2 as sm2
import numpy as np
import time


class GaussianClassification():
    def __init__(self, im): 
        self.im = im
        self.wl = np.asarray(im.bands.centers)
        self.imArr = im.Arr  
        self.nrows = im.Arr.shape[0]
        self.ncols = im.Arr.shape[1]
        self.nbands = im.Arr.shape[2]  
        
        
    def load_gt(self, fname, verbose=False):
        if verbose: 
            print('Loading groundtruth...')
            start_time = time.time()        
        print('Reading data code goes here.')
        if verbose: 
            print(f'Total time: {(time.time()-start_time):.2f} seconds')
    
    def train(self, tol=10**(-8), verbose=False):
        if verbose: 
            print('Training...')
            start_time = time.time()        
        print('Training code goes here.')        
        if verbose: 
            print(f'Total time: {(time.time()-start_time):.2f} seconds')
        
    def predict(self, verbose=False):
        if verbose: 
            print('Predicting...')
            start_time = time.time()        
        print('Predicting code goes here.')        
        if verbose: 
            print(f'Total time: {(time.time()-start_time):.2f} seconds')


    # Visualizations and Plots

    def plot_RGB(self, ):
        sm2.display_RGB(self.im.Arr, self.wl, rotate=True)
        plt.title('RGB Image');
        
    def plot_gt(self):
        plt.figure(figsize=(15,5)) 
        plt.imshow(np.flip(np.rot90(self.gt_im), axis=0), cmap='jet');
        plt.gca().invert_yaxis()  
        plt.xlabel('Row');
        plt.ylabel('Column');
        plt.title('Ground Truth Image')
    
    def plt_RGB_classes(self):
        class_only_image = np.zeros((self.nrows, self.ncols, self.nbands))
        for class_name in self.groundtruth_classes.keys():
            for x,y in self.groundtruth_classes[class_name]['locations']:
                class_only_image[int(x), int(y), :] = self.im.Arr[int(x), int(y), :]

        sm2.display_RGB(class_only_image, self.wl, stretch_pct=[0,99], rotate=True)
        plt.title('RGB Image');
        
    def plot_classification_results(self):
        plt.figure(figsize=(15,5)) 
        plt.imshow(np.flip(np.rot90(self.LDA_class_image), axis=0), cmap='jet');
        plt.gca().invert_yaxis()  
        plt.xlabel('Row');
        plt.ylabel('Column');
        plt.title('LDA Class Predictio n Image')
    
    def plt_class_means(self):
        plt.figure(figsize=(12,4))
        for i in range(self.nClasses):
            plt.plot(self.wl, self.class_means[i,:], label=self.class_names[i])
        plt.grid(True)
        plt.legend()
    
    def plot_class_sideBySide(self, fs=10):
        plt.figure(figsize=(fs,fs)) 
        plt.subplot(1,2,1)
        plt.imshow(self.gt_im, cmap='jet');
        plt.xlabel('Row');
        plt.ylabel('Column');
        plt.title('Ground Truth Image')
        plt.subplot(1,2,2)
        plt.imshow(self.LDA_class_image, cmap='jet');
        plt.xlabel('Row');
        plt.ylabel('Column');
        plt.title('LDA Class Predictio n Image')
        plt.tight_layout()
    
    def plt_scatter(self, b1=40, b2=150):
        # Create a scatterplot of the data
        plt.figure(figsize=(15,10))
        plt.scatter(self.im.List[:,b1], self.im.List[:,b2], s=5, alpha=0.5);
        plt.grid(True)
        plt.xlabel(f'Reflectance at {self.wl[b1]:.1f} nm')
        plt.ylabel(f'Reflectance at {self.wl[b2]:.1f} nm')
        plt.title('Scatterplot of Image Spectra');
        
    def plt_scatter_gt(self, b1=40, b2=150):
        # Create a scatterplot of the data, colored by ground truth class
        plt.figure(figsize=(15,10))
        bk_indiexes = np.where(self.gt_list==0)[0] # get hte indices only for pixels in ground truth classes
        plt.scatter(self.im.List[bk_indiexes,b1], self.im.List[bk_indiexes,b2], c='grey', s=3, alpha=0.5);
        gt_indiexes = np.where(self.gt_list>0)[0] # get hte indices only for pixels in ground truth classes
        plt.scatter(self.im.List[gt_indiexes,b1], self.im.List[gt_indiexes,b2], c=self.gt_list[gt_indiexes], s=3, alpha=0.5, cmap='jet');
        plt.grid(True)
        plt.xlabel(f'Reflectance at {self.wl[b1]:.1f} nm')
        plt.ylabel(f'Reflectance at {self.wl[b2]:.1f} nm')
        plt.title('Scatterplot of Image Spectra');
    
        
        
        
        
        
        