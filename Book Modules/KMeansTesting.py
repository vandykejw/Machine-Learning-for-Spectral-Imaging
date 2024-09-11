import numpy as np
import copy

def KMeans(imArr, k, num_iterations=15, dist_type='Correlation'):
    '''
    K Means function using Euclidean or Correlation distance.
    '''
    print(f'Running Kmeans, k={k}, num_iterations={num_iterations}, and using {dist_type} distance.')
    
    # Compute parameters
    nRows, nCols, nBands = imArr.shape
    nPix = nRows*nCols
    imList = np.reshape(imArr, (nRows*nCols, nBands))

    # Step 0:
    class_index = np.random.randint(low=0, high=k, size=nPix)
    
    # Create a array of normalized (subratct mean, divide by standard deviation) spectra
    #   - only used for correlation metric
    if dist_type == 'Correlation':
        imListNormed = copy.copy(imList)
        imListNormed.shape
        imList_mu = np.mean(imList, axis=1)
        for j in range(nBands):
            imListNormed[:,j] = imListNormed[:,j] - imList_mu
        imList_stdev = np.std(imList, axis=1)
        for j in range(nBands):
            imListNormed[:,j] = imListNormed[:,j] / imList_stdev
    
    # Repeat num_iterations times:
    print(f"Iteration =", end =" ")
    for iteration in range(num_iterations):
        print(f" {iteration}", end =" ")
        # Step 1: Compute Class Means
        class_means = np.zeros((k,nBands))
        for i in range(k):
            idx = np.where(class_index==i)
            spectra_for_this_class = np.squeeze(imList[idx,:])
            class_means[i,:] = np.mean(spectra_for_this_class, axis=0)

        # Step 2: Re-assign Class Labels
        if dist_type == 'Euclidean':
            
            distance_to_mean = np.zeros((nPix,k))
            for i_cls in range(k):    
                imListTemp = copy.copy(imList)
                for j in range(nBands):
                    imListTemp[:,j] = imListTemp[:,j] - class_means[i_cls,j]
                distance_to_mean[:,i_cls] = np.sum(imListTemp**2, axis=1)
            class_index = np.argmin(distance_to_mean, axis=1)
            
        else:
            
            correlation_to_mean = np.zeros((nPix,k))
            for i_cls in range(k): 
                class_mean = class_means[i_cls,:]
                class_mean_Normed = (class_mean - np.mean(class_mean)) / np.std(class_mean)
                correlation_to_mean[:,i_cls] = (1/nBands)*np.matmul(imListNormed,class_mean_Normed)
            class_index = np.argmax(correlation_to_mean, axis=1)

        class_image = np.reshape(class_index, (nRows,nCols))
    print(" ") # Just to start a newline on the printing
    
    return class_index, class_image
    

    