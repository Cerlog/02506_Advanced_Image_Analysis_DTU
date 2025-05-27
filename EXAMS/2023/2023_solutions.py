'''
Correct answers:
1  (g) 
2  (d)
3  (i)
4  (g)
5  (f)
6  (d)
7  (e)
8  (f)
9  (b)
10 (g)
11 (h)
12 (d)
13 (h)
14 (c)
15 (i)
16 (c) 
17 (a)
18 (e)
19 (e)
20 (c)
'''


# %%
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import scipy.ndimage
data_dir = 'DATA/'


#%% 
# QUESTION 1, Gaussian filtering

# Read images
im = skimage.io.imread(f'{data_dir}wood.png').astype(np.float32)
L_gt = skimage.io.imread(f'{data_dir}wood_LoG.tif')

# Gaussian filters from exercise week 1
def gaussians(s):
    x = np.arange(-np.ceil(4*s), np.ceil(4*s)+1)
    x = np.reshape(x, (-1,1))
    g = 1/(s*np.sqrt(2*np.pi))*np.exp(-x**2/(2*s**2))
    dg = -x/(s**2)*g
    ddg = -x/(s**2)*dg - 1/(s**2)*g
    return g, dg, ddg

# Gaussian parameter sigma
s = 5.5
g, dg, ddg = gaussians(s)

# Second order Gaussian derivative in x and y directions
Lxx = scipy.ndimage.convolve(scipy.ndimage.convolve(im, ddg), g.T)
Lyy = scipy.ndimage.convolve(scipy.ndimage.convolve(im, ddg.T), g)

# Laplacian of Gaussian
L = Lxx + Lyy


fig, ax = plt.subplots(2,2)
ax[0][0].imshow(im)
ax[0][1].imshow(L)
ax[1][0].imshow(L_gt)
ax[1][1].imshow(L-L_gt)
fig.suptitle(f'The correct answer is {s}')
plt.show()



#%% 
# QUESTION 2, Mean bending 
bended = np.loadtxt(data_dir + 'zealand.txt')
fig, ax = plt.subplots(figsize=[10, 5])
ax.plot(bended[:, 0], bended[:, 1], 'r.')
ax.set_aspect('equal')
b = 2 * bended - np.roll(bended, 1, axis=0) - np.roll(bended, -1, axis=0)
b = np.sqrt((b**2).sum(axis=1)).mean()
ax.set_title(b)
plt.show()


#%% 
# QUESTION 3, Blob detection

# Compute Laplacian of Gaussians and find local maxima (dark blobs) and 
# minima (bright blobs)

im = skimage.io.imread(f'{data_dir}fiber.png').astype(float)

# Gaussian filters from exercise week 1
def gaussians(s):
    x = np.arange(-np.ceil(4*s), np.ceil(4*s)+1)
    x = np.reshape(x, (-1,1))
    g = 1/(s*np.sqrt(2*np.pi))*np.exp(-x**2/(2*s**2))
    dg = -x/(s**2)*g
    ddg = -x/(s**2)*dg - 1/(s**2)*g
    return g, dg, ddg

im_sp = []

s = 3
g, dg, ddg = gaussians(s)
Lxx = scipy.ndimage.convolve(scipy.ndimage.convolve(im, ddg), g.T)
Lyy = scipy.ndimage.convolve(scipy.ndimage.convolve(im, ddg.T), g)
L = Lxx + Lyy

# Own implementation of local maxima and minima
def get_local_extrema_2D(L):
    r,c = L.shape
    ct_im = np.zeros((r,c))
    for i in range(-1,2):
        for j in range(-1,2):
            to_row = r - 1 + i
            to_col = c - 1 + j
            ct_im[1:-1,1:-1] += (L[1:-1,1:-1] > L[1+i:to_row,1+j:to_col])
    coord_dark = np.where(ct_im[1:-1,1:-1]==8)
    coord_bright = np.where(ct_im[1:-1,1:-1]==0)
    return coord_dark, coord_bright

coord_dark, coord_bright = get_local_extrema_2D(L)

# Display (not needed for computing the solution)
fig, ax = plt.subplots()
ax.imshow(im, cmap='gray')
ax.plot(coord_dark[1], coord_dark[0], '.r')
ax.plot(coord_bright[1], coord_bright[0], '.b')
ax.set_title(f'Dark blobs: {len(coord_dark[0])} Bright blobs: {len(coord_bright[0])}')
plt.show()

#%% 
# QUESTION 4, Blob size

# The solution is to find all pixels where the blob response is larger 
# than 10, because we obtain positive response from dark blobs.

# Read in the image for displaying the result
im = skimage.io.imread(f'{data_dir}cement.png').astype(float)

# Read in the data
data_in = np.loadtxt(f'{data_dir}cement.txt')
blob_resp = data_in[:,0]
coords = data_in[:,1:3]
magnitude = data_in[:,3]

# Find data where blob response is larger than 10
idx = np.where(blob_resp > 10)
coords_1 = coords[idx]
magnitude_1 = magnitude[idx]

# Compute the average diameter as two times the radius
d = (2*np.sqrt(2*magnitude_1)).mean()

# Display (not needed for computing the solution)
n_theta = 100
rad = np.arange(0,2*np.pi, 2*np.pi/n_theta)
rad = np.append(rad, 2*np.pi)
cx = np.sin(rad)
cy = np.cos(rad)

fig, ax = plt.subplots()
ax.imshow(im, cmap='gray')
ax.plot(coords[:,1], coords[:,0], '.r', alpha=0.75)
for i, (c, m) in enumerate(zip(coords, magnitude)):
    ax.plot(cx*np.sqrt(2*m)+c[1], cy*np.sqrt(2*m)+c[0],'r', alpha=0.75)
ax.set_title(f'The solution is {d:0.1f}')
plt.show()



#%% 
# QUESTION 5, Feature-based segmentation

# Per pixel probabilities are given, and the task is to compute the 
# probabilities for each pixel in the image, and count how many pixels 
# that has the label 3 relative to the total number of pixels in the image. 
# Note that there were a little typo in the list of answers that says 0.408.
# The correct answer is 0.410.

# Read in data
assignment_image = skimage.io.imread(f'{data_dir}liver_assignment.png')
prob_dict = np.loadtxt(f'{data_dir}dictionary_probabilities.txt')

# Build the probaility image for display and for computing the number of 
# pixels with label 3
prob_im = prob_dict[assignment_image]

# Segmentation image by finding the label with the heighest value in each pixel
seg_im = np.argmax(prob_im,axis=2)

# Display (not needed for computing the solution)
fig, ax = plt.subplots(1,2)
ax[0].imshow(prob_im)
ax[1].imshow(seg_im)
fig.suptitle(f'The correct answer is: {(seg_im==2).sum()/(assignment_image.size):0.3f}')
plt.show()


# #%% Simple solution with a for loop

# n_label = 0

# for i in np.ravel(assignment_image):
#     if np.argmax(prob_dict[i]) == 2:
#         n_label += 1
# print(f'The correct answer is: {n_label/assignment_image.size:0.3f}')


#%% 
# QUESTION 6, Gaussian features
'''
The Gaussian features that we use for segmentation are described 
correctly in answer (d). This can be seen from Chapter 3 in the lecture
note.
'''

#%% 
# QUESTION 7, Point correspondance

# Load the point sets
q = np.loadtxt(f'{data_dir}points_q.txt').T
p = np.loadtxt(f'{data_dir}points_p.txt').T

# Dispaly the points and their correspondance (not needed for computing the solution)
fig, ax = plt.subplots()
ax.plot(q[0], q[1], 'm.')
ax.plot(p[0], p[1], 'c.')
ax.plot(np.vstack((q[0], p[0])), np.vstack((q[1], p[1])), 'k-', linewidth=0.5)
plt.show()

# Compute mean points and distance to mean points for estimating the scale
mu_q = q.mean(axis=1, keepdims=True)
mu_p = p.mean(axis=1, keepdims=True)
d_q = np.sqrt(((q-mu_q)**2).sum(axis=0))
d_p = np.sqrt(((p-mu_p)**2).sum(axis=0))

# Estimated scale
s_est = d_q.sum()/d_p.sum()

# Covariance matrix
C = (q - mu_q)@(p - mu_p).T
U,S,V = np.linalg.svd(C)
R_est = U@V
theta_est = np.arccos(R_est[0,0])
angle_est = 360*theta_est/(2*np.pi)

# Transformation
t_est = (q - s_est*R_est@p).mean(axis=1, keepdims=True)

# Check result by plotting  (not needed for computing the solution)
q_est = s_est*R_est@p + t_est

fig, ax = plt.subplots()
ax.plot(q_est[0], q_est[1], 'm.')
ax.plot(q[0], q[1], 'c.')
ax.plot(np.vstack((q_est[0], q[0])), np.vstack((q_est[1], q[1])), 'k-', linewidth=0.5)
ax.set_title(f'Scale {s_est:0.3f}, Angle {angle_est:0.2f}, Translation [{t_est[0][0]:0.2f}, {t_est[1][0]:0.2f}]')
plt.show()


#%% 
# QUESTION 8, SIFT matching

# Load the SIFT features
sift_a = np.loadtxt(f'{data_dir}sift_a.txt')
sift_b = np.loadtxt(f'{data_dir}sift_b.txt')

coord_a = sift_a[:,:2]
coord_b = sift_b[:,:2]
desc_a = sift_a[:,2:]
desc_b = sift_b[:,2:]

# Create a list of translations
t = []

# Compute distance between one descriptor in set a and all in set b
for i, desc in enumerate(desc_a):
    d = np.linalg.norm((desc - desc_b), axis=1)
    idx = np.argsort(d)
    if d[idx[0]]/d[idx[1]] < 0.8:
        t.append(coord_a[i] - coord_b[idx[0]]) 

# Make array of translations and compute the mean
t = np.vstack(t)
tm = t.mean(axis=0)
print(f'The solution is {np.linalg.norm(tm):0.2f}')


#%% 
# QUESTION 9, MRF prior
''' Option (a) is not correct, as it describes likelihood, not prior. 
Option (b) is correct.
Option (c) is not correct, the first sentence contributes no information, and 
the second sentence describes posterior.
Option (d) is not correct. The last sentence describes some type of likelihood.
'''

#%%
# QUESTION 10 Energy change
vals = np.array([[7, 7, 8], [2, 2, 2]])
E_old = ((vals - np.array([[8], [2]]))**2).sum() + 8 * 16 + 6 * 12 
E_new = ((vals - np.array([[2], [8]]))**2).sum() + 2 * 12
change = E_new-E_old
print(change)


#%% 
# QUESTION 11, External force 
I = skimage.io.imread(data_dir + 'flower.png')
snake = np.loadtxt(data_dir + 'snake.txt').astype(int)

fig, ax = plt.subplots(figsize=[10, 5])
ax.imshow(I, cmap=plt.cm.gray)
ax.plot(snake[:, 1], snake[:, 0], 'r.')
ax.plot(snake[0, 1], snake[0, 0], 'ro')

mask = skimage.draw.polygon2mask(I.shape, snake)
ax.imshow(mask, alpha=0.5)
m_in = np.mean(I[mask])
m_out = np.mean(I[~mask])

val = I[snake[0, 0], snake[0, 1]]
force = (m_in - m_out) * (2 * val - (m_in + m_out))
ax.set_title(force)
plt.show()


#%% QUESTION 12, Beta effect
''' When increasing beta, the result should be smoother, eliminating options (c) 
and (e). Increasing beta should not make curve longer, eliminating options (a) 
and (f). Increasing beta should lead to even smoother curve, but without 
shrinkage, eliminating option (b). The correct answer is (d) where curve is 
smoother but not shrunk.
'''


#%%
# QUESTION 13, Surface cost
im = skimage.io.imread(data_dir + 'lines.png')
max_delta = 3
lines = np.loadtxt(data_dir + 'lines.txt').astype(int)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im, cmap='gray')
ax.plot(lines.T, linewidth=3)

x = np.arange(im.shape[1])
costs = [im[lines[i, :], x].sum() for i in range(lines.shape[0])] 
costs = np.array(costs).astype(float)
difs = np.max(np.abs(lines[:, :-1] - lines[:, 1:]), axis=1)
idx = np.argmin(costs + (costs.max())*(difs > max_delta))
ax.set_title(idx)
plt.show()


#%% QUESTION 14, Surface detection
''' Smoothness parameter DeltaX has to be larger or equal to 1, eliminating
option (b). If parameters (a) were used, the surface would in the first colunm
pass pixel with value 2, not 6. If parameters (d) were used, the surface would
in the first column pass pixel with value 5, not 6. If parameters (e) were used, 
the surface would in the sixth column pass pixel with value 6, not 5. This
leaves us with the correct solution (c).
'''


#%% QUESTION 15, Trained MLP
def predict(x1, x2):
    h2 = max(10 * x2 - 9.1 * x1 + 0.3, 0.0)
    h1 = max(-9.1 * x2 + 9 * x1 + 0.7, 0.0)
    y2 = 5.4 * h2 - 4.6 * h1 - 0.5
    y1 = -5.9 * h2 + 5.6 * h1 + 0.2    
    return((y2 > y1) + 1)

fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
for x1 in np.linspace(-1, 1, 15, endpoint=True):
    for x2 in np.linspace(-1, 1, 15, endpoint=True):
        ax.text(x1, x2, predict(x1, x2))
plt.show()


#%% 
# QUESTION 16, Partial derivatives
''' We have L = - sum_i t_i * ln(y_i) and we need to compute dL/dY_i.
For a partial derivative in respect to y_i, only the summsyion term containing 
y_i is  relevant, that is the term - t_i * ln(y_i).
Factor t_i is a constant and  ln'(x)=1/abs(x). So, we have dL/dY_i= - t_i/y_i.   
'''
y = np.array([0.05, 0.2, 0.4, 0.1, 0.25])
t = np.array([0, 1, 0, 0, 0])
print(-t/y)


#%% QUESTION 17, Momentum
'''
The momentum is described in the lecture note Chapter 9. The correct answer
is (a)
''' 

#%% 
# QUESTION 18, Classification training
'''We expect training loss to always be smaller than validation loss, and the 
training accuracy larger than validation accuracy -- that eliminates options (d)
and (f). We expect the training performance to keep improving so option (a) is 
eliminated. The validation performance may stop impoving or even get worse (due
to overfitting), and validation performance will not keep improving while 
training performance stagnates. This eliminates options (b) and (c). Left is the
correct solution (e) which shows slight overfitting after epoch 50.'''


# %%
# QUESTION 19, Bottleneck size
dims = np.array([268, 352])
down1 = (dims - 4 - 4)/2  # after two conv5x5 layers and moxpool2x2
down2 = (down1 - 4 - 4)/2 # after second block of two conv5x5 layers and moxpool2x2
bottom = (down2 - 4) # just one conv in the third block
print(bottom)

# #%% If wanting to test with torch
# import torch
# class exam_model(torch.nn.Module):
#    
#     def __init__(self, out_channels=2):
#         '''Used only for testing spatial dimensions, so chanels=1, and no activation.'''
#         super().__init__()
#         self.conv = torch.nn.Conv2d(1, 1, 5)  
#         self.pool = torch.nn.MaxPool2d(2, 2)        
#       
#     def forward(self, x):
#         l1 = self.conv(self.conv(x))
#         l2 = self.conv(self.conv(self.pool(l1)))
#         out = self.conv(self.pool(l2))
#         return out
#    
# model = exam_model()
# image_in = torch.rand(size=(1, 1, 268, 352), dtype=torch.float32)
# logits = model(image_in)
# print(image_in.shape)
# print(logits.shape)


# %%
# QUESTION 20, Data augmentation
'''Data augmentation is not a technique for sampling or organizing data samples.
The correct answer is (c).
'''

# %%
