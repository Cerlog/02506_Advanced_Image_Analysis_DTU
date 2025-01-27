#%%

import skimage.io
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

#%%

def segmentation_energy(S, D, mu, beta):
    # TODO -- add your code here
    
    # likelihood energy
    U1 = 0
    
    # prior energy
    U2 = 0
    
    return U1, U2

def segmentation_histogram(ax, D, S, edges=None):
    '''
    Plot histogram for grayscale data and each segmentation label.
    '''
    if edges is None:
        edges = np.linspace(D.min(), D.max(), 100)
    ax.hist(D.ravel(), bins=edges, color = 'k')
    centers = 0.5 * (edges[:-1] + edges[1:])
    for k in range(S.max() + 1):
        ax.plot(centers, np.histogram(D[S==k].ravel(), edges)[0])
        

path = '../data/week5/'
D = skimage.io.imread(path + 'noisy_circles.png').astype(float)

# Ground-truth segmentation.
GT = skimage.io.imread(path + 'noise_free_circles.png')
(mu, S_gt) = np.unique(GT, return_inverse=True)
S_gt = S_gt.reshape(D.shape)

segmentations = [S_gt]  # list where I'll place different segmentations

#%% Find some configurations (segmentations) using conventional methods.

# Simple thresholding
S_t = np.zeros(D.shape, dtype=int) + (D > 100) + (D > 160) # thresholded
segmentations += [S_t]

# Gaussian filtering followed by thresholding
D_s = scipy.ndimage.gaussian_filter(D, sigma=1, truncate=3, mode='nearest')
S_g = np.zeros(D.shape, dtype=int) + (D_s > 100) + (D_s > 160) 
segmentations += [S_g]

# Median filtering followed by thresholding
D_m = scipy.ndimage.median_filter(D, size=(5, 5), mode='reflect')
S_t = np.zeros(D.shape, dtype=int) + (D_m > 100) + (D_m > 160) # thresholded
segmentations += [S_t]


#%% visualization
fig, ax = plt.subplots()
ax.imshow(D, vmin=0, vmax=255, cmap=plt.cm.gray)
plt.show()


fig, ax = plt.subplots(3, len(segmentations), figsize=(10, 10))
beta = 100

for i, s in enumerate(segmentations):
    ax[0][i].imshow(s)

    V1, V2 = segmentation_energy(s, D, mu, beta)
    ax[0][i].set_title(f'likelihood: {V1:.2g}\nprior: {V2}\nposterior: {V1+V2:.2g}')
    
    segmentation_histogram(ax[1][i], D, s)
    ax[1][i].set_xlabel('Intensity')
    ax[1][i].set_ylabel('Count')
    
    err = S_gt - s
    ax[2][i].imshow(err, vmin=-2, vmax=2, cmap=plt.cm.bwr)
    ax[2][i].set_title(f'Pixel error: {(err != 0).sum()}')

fig.tight_layout()
plt.show()   


# %%
