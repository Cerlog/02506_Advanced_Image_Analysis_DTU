"""
Helping functions for extracting features from images

Vedrana Andersen Dahl (vand@dtu.dk) 
Anders Bjorholm Dahl (abda@dtu.dk)
"""

#%%
import numpy as np
import scipy.ndimage


######################################################################################
######################################################################################
#                       GAUSSIAN DERIVATIVE FEATURES
######################################################################################
######################################################################################
def get_gauss_feat_im(im, sigma=1, normalize=True):
    """Gauss derivative features for every image pixel.
    
    This function applies Gaussian filters with different derivative orders
    to extract a multi-dimensional feature representation at each pixel
    of a 2D image. The derivatives capture various image properties such as
    edges, corners, and texture gradients.
    It generates 15 different feature channels by applying Gaussian derivatives
    of orders 0 to 4 in both x and y directions.
    
    Arguments:
        im: a 2D image, shape (r, c).
        sigma: standard deviation for the Gaussian filter defining the scale.
        normalize: flag to indicate if the resulting features should be
                   normalized to have zero-mean and unit variance.
    Returns:
        imfeat: 3D array of size (r, c, 15) where each pixel is represented by a 
                15-dimensional feature vector computed from different orders of 
                Gaussian derivatives.
    Author: vand@dtu.dk, 2022
    """
    # Define the list of derivative orders to compute.
    # Orders include the 0th derivative (smoothed image) and higher order derivatives.
    im = im.astype(float)
    orders = [0, 
              [0, 1], [1, 0], 
              [0, 2], [1, 1], [2, 0], 
              [0, 3], [1, 2], [2, 1], [3, 0],
              [0, 4], [1, 3], [2, 2], [3, 1], [4, 0]]
    
    # Apply the Gaussian filter with each derivative order.
    # For each order in the list, the function scipy.ndimage.gaussian_filter computes
    # the Gaussian derivative response of the image.
    imfeat = [scipy.ndimage.gaussian_filter(im, sigma, o) for o in orders]    
    
    # Stack the individual feature images along the third dimension.
    # The resulting array has shape (r, c, 15) corresponding to the 15 derivative responses.
    imfeat = np.stack(imfeat, axis=2)
   
    if normalize:
        # Subtract the mean of each feature (across all pixels) to center the feature values around zero.
        imfeat -= np.mean(imfeat, axis=(0, 1))
        # Calculate the standard deviation for each feature.
        std = np.std(imfeat, axis=(0, 1))
        # Avoid division by zero by setting any zero standard deviation to one.
        std[std == 0] = 1
        # Normalize the features so that each has unit variance.
        imfeat /= std
    
    return imfeat

def get_gauss_feat_multi(im, sigmas=[1, 2, 4], normalize=True):
    '''Multi-scale Gauss derivative features for every image pixel.
    
    This function computes Gaussian derivative features over multiple scales.
    For each sigma in the list, it extracts 15 feature channels using the 
    get_gauss_feat_im function, reshapes them into a 2D array, and then 
    aggregates them into a single 3D array with dimensions corresponding to 
    the number of pixels, number of scales, and 15 derivative features.
    
    Arguments:
        im: a 2D image, shape (r, c).
        sigmas: list of standard deviations for Gaussian derivatives.
        normalize: flag indicating if each feature should be normalized.
    
    Returns:
        imfeats: a 3D array of size (num_pixels, n_scale, 15). Each row 
                corresponds to a pixel's feature vector computed at each scale.
    
    Author: abda@dtu.dk, 2021
    '''
    imfeats = []  # Initialize a list to store features for each scale

    for s in sigmas:
        # Compute the Gaussian derivative features for the current sigma
        feat = get_gauss_feat_im(im, s, normalize)
        # Reshape each feature image from shape (r, c, 15) into (num_pixels, 15)
        imfeats.append(feat.reshape(-1, feat.shape[2]))
    
    # Convert the list of feature arrays to a NumPy array.
    # The shape is initially (n_scale, num_pixels, 15) and we are transposing it
    # to have the shape (num_pixels, n_scale, 15) fo
    
    # Return the multi-scale feature representation.
    #return imfeatsr  #easier pixel-wise processing.
    imfeats = np.asarray(imfeats).transpose(1, 0, 2)
    return imfeats



######################################################################################
######################################################################################
#                       PATCH-BASED FEATURES
######################################################################################
######################################################################################

def im2col(im, patch_size=[3, 3], stepsize=1):
    """Rearrange image patches into columns
    Arguments:
        image: a 2D image, shape (r,c).
        patch size: size of extracted paches.
        stepsize: patch step size.
    Returns:
        patches: a 2D array which in every column has a patch associated 
            with one image pixel. For stepsize 1, number of returned column 
            is (r-patch_size[0]+1)*(c-patch_size[0]+1) due to bounary. The 
            length of columns is pathc_size[0]*patch_size[1].
    """
    
    r, c = im.shape
    s0, s1 = im.strides    
    nrows =r - patch_size[0] + 1
    ncols = c - patch_size[1] + 1
    shp = patch_size[0], patch_size[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(im, shape=shp, strides=strd)
    return out_view.reshape(patch_size[0]*patch_size[1], -1)[:, ::stepsize]


def ndim2col(im, block_size=[3, 3], stepsize=1):
    """Rearrange image blocks into columns for N-D image (e.g. RGB image)

    This function converts an image into column vectors of its blocks (patches). For a
    grayscale (2D) image, it directly calls im2col. For a multi-channel (e.g. RGB)
    image, it processes each channel separately and concatenates the patches vertically.

    Arguments:
        im: Input image, which can be either a 2D grayscale image or a 3D multi-channel image.
        block_size: The size of each block (patch) to extract, given as [height, width].
        stepsize: The number of pixels to step when sliding the block across the image.

    Returns:
        patches: A 2D array where each column is a flattened block from the image. For multi-channel images,
                 the patches from each channel are concatenated in order.
    """
    # If the image is 2D (grayscale), use the existing im2col function.
    if im.ndim == 2:
        return im2col(im, block_size, stepsize)
    else:
        # For a multi-channel image assume shape is (rows, columns, channels)
        r, c, l = im.shape

        # Calculate the number of patches per channel.
        num_patches = (r - block_size[0] + 1) * (c - block_size[1] + 1)
        # Each patch is flattened into a vector with 'patch_elements' elements.
        patch_elements = block_size[0] * block_size[1]

        # Initialize an array to hold the concatenated patches from all channels.
        # The output shape is (l * patch_elements) rows by (num_patches) columns.
        patches = np.zeros((l * patch_elements, num_patches))

        # Iterate over each channel
        for i in range(l):
            # Determine the row indices in the output array corresponding to the current channel.
            start = i * patch_elements
            end = (i + 1) * patch_elements

            # Extract patches from the i-th channel using im2col
            # These patches are arranged in a 2D array where each column is a flattened block.
            patches[start:end, :] = im2col(im[:, :, i], block_size, stepsize)
        return patches

#%%
import skimage.io
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    #%% features based on gaussian derivatives
    import skimage.io
    import matplotlib.pyplot as plt
    filename = '3labels/training_image.png'
    I = skimage.io.imread(filename).astype(float)/255
    I = I[200:400, 200:400] # smaller image such that we can see 
    fig, ax = plt.subplots()
    ax.imshow(I)
    
    sigma = 5
    gf = get_gauss_feat_im(I, sigma)
    
    fig,ax = plt.subplots(5, 3, figsize=(15, 25))


    ax = ax.ravel()
    for i, a in enumerate(ax):
        a.imshow(gf[..., i], cmap='jet')
        a.set_title(f'layer {i}')
    plt.show()
            
            
    #%% features based on image patches
    I = skimage.io.imread(filename).astype(float)/255
    I = I[300:320, 400:420] # smaller image such that we can see 
    fig, ax = plt.subplots()
    ax.imshow(I)

    pf = im2col(I, [3, 3])
    pf = pf.reshape((9, I.shape[0]-2, I.shape[1]-2))
            
    fig,ax = plt.subplots(3,3)
    for j in range(3):
        for i in range(3):
            ax[i][j].imshow(pf[3*i+j], cmap='jet')
            ax[i][j].set_title(f'layer {3*i+j}')
            
    plt.show()
    