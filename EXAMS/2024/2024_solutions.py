'''
Correct answers:
1 (c)
2 (g)
3 (i)
4 (b)
5 (c)
6 (d)
7 (j)
8 (c) 
9 (f)
10 (i)
11 (d)
12 (f)
13 (d)
14 (j)
15 (i)
16 (b) 
17 (i)
18 (b)
19 (i)
20 (b)
'''


# %%
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
%matplotlib qt

data_dir = 'DATA/'


#%% 
# QUESTION 1, Gaussian filterng

'''By looking ath the filtered image we can conclude that filtering 
needs to be the same in the horizontal and the vertical direction. Filtering with 
the first order derivative would not have symmetrical outcome. We can than
pick answer (c).'''

# #%% Optional: Q1 Solution based on trying all options. 
# img = skimage.io.imread(data_dir + 'lung/lung.png').astype(float)
# img_ref = skimage.io.imread(data_dir + 'lung/lung_processed.tif')

# fig, ax = plt.subplots(1,2)
# ax[0].imshow(img, cmap='gray')
# ax[1].imshow(img_ref, cmap='gray')

# def gaussians(s):
#     x = np.arange(-np.ceil(4*s), np.ceil(4*s)+1)
#     x = np.reshape(x, (-1,1))
#     g = 1/(s*np.sqrt(2*np.pi))*np.exp(-x**2/(2*s**2))
#     dg = -x/(s**2)*g
#     ddg = -x/(s**2)*dg - 1/(s**2)*g
#     return g, dg, ddg

# img_options={}

# g, dg, ddg = gaussians(10)
# img_options['a'] = scipy.ndimage.convolve(img, dg.T)

# g, dg, ddg = gaussians(np.sqrt(10))
# img_options['b'] = scipy.ndimage.convolve(scipy.ndimage.convolve(img, g.T), dg)
# g, dg, ddg = gaussians(4)
# Lxx = scipy.ndimage.convolve(scipy.ndimage.convolve(img, ddg), g.T)
# Lyy = scipy.ndimage.convolve(scipy.ndimage.convolve(img, ddg.T), g)
# img_options['c'] = Lxx + Lyy

# g, dg, ddg = gaussians(7)
# img_options['d'] = scipy.ndimage.convolve(scipy.ndimage.convolve(img, dg), dg.T)

# best_value = np.inf
# fig, ax = plt.subplots(2, 4)
# for i, a in zip(img_options, ax.T):
#     a[0].imshow(img_options[i], cmap='gray')
#     a[0].set_title('result ' + i)
#     diff = img_options[i] - img_ref
#     a[1].imshow(diff, cmap='gray')
#     val = np.abs(diff).sum()
#     a[1].set_title(f'difference\n{val:.0}')

#%%
#  QUESTION 2, Image smoothness

def smoothness(I):
    I = I.astype(int)  # important before subtracting
    h = np.minimum((I[:, 1:] - I[:, :-1])**2, 100).sum()
    v = np.minimum((I[1:, :] - I[:-1, :])**2, 100).sum()
    return h + v

images = [skimage.io.imread(data_dir + 'slice/slice' + e + '.png') 
          for e in ['', '_G', '_M']]
fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
for a, I in zip(ax, images):
    a.imshow(I, cmap='gray', vmin=0, vmax=255)
    a.set_title(smoothness(I))
plt.show()

#%% 
# QUESTION 3, Blob detection

im = skimage.io.imread(data_dir + '/narwhale/narwhale.png')

# Multiply with t for scale normalization, only scales around 6 are of interest
sc_vol = [t * skimage.io.imread(data_dir + 
                f'narwhale/narwhale_scale_space_{t}.tif') for t in [4, 6, 8]] 

ct_im = np.zeros(im.shape)
for t in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        for k in [-1, 0, 1]:
            ct_im[1:-1, 1:-1] += (sc_vol[1][1:-1, 1:-1] > 
                    sc_vol[1+t][1+j:im.shape[0]-1+j, 1+k:im.shape[1]-1+k])

r, c = np.where((ct_im==26) & (im < 100))  
plt.imshow(im)
plt.plot(c, r, 'r.')
plt.title(f'Number blobs: {len(r)}')
plt.show()

#%% 
# QUESTION 4, Gaussian scale space
'''The image (B) seems sharpest, so options starting with (H) or (F) may be 
eliminated. The image (F) is not the most blury, so option (a) may be eliminated.
To choose between option (b) and (e) we inspect a pair (G) and (C), a pair (F) 
and (E) and a pair (D) and (A). In all three pairs, the first image is sharper. 
So order is B, G, C, F, E, D, A, H and (b) is the correct answer.
'''
#%% 
# QUESTION 5, Feature clustering
'''Number of dictionary elements (clusters) is typically much larger than number 
of labels because the local appearance (texture) may vary significantly. We can
conclude that (c) is the correct answer.'''

#%% 
# QUESTION 6, Feature-based segmentation

im = skimage.io.imread(data_dir + 'bean/bean_fibers.png')
label_im = skimage.io.imread(data_dir + 'bean/bean_fibers_labels.png')
assignment_im = skimage.io.imread(data_dir + 'bean/bean_assignment_image.png')

fig, ax = plt.subplots(1,3)
ax[0].imshow(im)
ax[1].imshow(label_im)
ax[2].imshow(assignment_im)

answer = []
for a in [17, 27, 117, 212]:
    r, c = np.where(assignment_im==a)
    p = [(label_im[r, c] == l).sum() for l in [0, 127, 255]]
    idx = np.argmax(np.array(p))
    print(f'Cluster {a}: label {idx+1}')
    answer.append(idx+1)
fig.suptitle(answer)
plt.show()
    


#%% 
# QUESTION 7, Matching blobs

p_in = np.loadtxt(data_dir + 'blobs/p_xyd.txt')
p = p_in[:, :2].T
dp = p_in[:, 2]
q_in = np.loadtxt(data_dir + 'blobs/q_xyd.txt')
q = q_in[:, :2].T
dq = q_in[:, 2]

theta = 48.3 * np.pi/180
R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
s = 0.84
t = np.array([[-10.3, 61.78]]).T
phat = s * R @ q + t
dphat = s * dq  # scaling the radius of the transformed blobs

distance = np.sqrt(((p - phat) ** 2).sum(axis=0))
r = dp/dphat
match = (distance < 2) & (0.8 < r) & (r < 1.25)

fig, ax = plt.subplots()
colors = {True: 'g', False: 'r'}
for i in range(len(match)):
    ax.add_patch(plt.Circle([p[0, i], p[1, i]], dp[i]/2, color='b', fill=False))
    ax.add_patch(plt.Circle([phat[0, i], phat[1, i]], dphat[i]/2, 
                            color = colors[match[i]], fill=False))
ax.set_xlim((0, 120))
ax.set_ylim((0, 120))
ax.axis('equal')
ax.set_title(f'{match.sum()} matches')
plt.show()

#%% 
# QUESTION 8, SIFT feature matching
'''Blobs are detected at a certain location and a certain scale. All other computation
is done in respect to this location and the scale, making SIFT features invariant
to translation and to scale. Blob detection is not used for subpixel precision, 
finding correspondence or computing the SIFT descriptor.'''

#%% 
# QUESTION 9, Segmentation energy

I = skimage.io.imread(data_dir + 'nerves/nerves.png').astype(float)/255
S = skimage.io.imread(data_dir + 'nerves/nerves_segmentation.png')>0

prior = ((I[S] - 0.6)**2).sum() + ((I[~S] - 0.2)**2).sum()
posterior = 0.1 * (S[1:] != S[:-1]).sum() + 0.05 * (S[:, 1:] != S[:, :-1]).sum()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(I, vmin=0, vmax=1, cmap=plt.cm.gray)
ax[1].imshow(S)
ax[1].set_title(prior + posterior)
plt.show()

# #%% ...IF SOMEBODY DECIDES TO USE MAXFLOW
# import maxflow
# g = maxflow.Graph[float]()
# nodeids = g.add_grid_nodes(I.shape)
# o = np.ones(shape = nodeids.shape)
# g.add_edges(nodeids[:-1, :], nodeids[1:, :], 0.1 * o[:-1, :], 0.1 * o[:-1, :])
# g.add_edges(nodeids[:, :-1], nodeids[:, 1:], 0.05 * o[:, :-1], 0.05 * o[:, :-1])
# g.add_grid_tedges(nodeids, (I - 0.6)**2, (I - 0.2)**2)
# flow = g.maxflow()
# print(f'Flow: {flow}, difference: {((g.get_grid_segments(nodeids) > 0)!=S ).sum()}')

#%% 
# QUESTION 10, Parameter range
# It is enough to consider two pixels, one with value 9 in the left part of the
# image and one with value 2 in the right part of the image.
red = 2
gray = 8
# Beta needs to be large enough that pixel with value 2 in the left part is gray
range_start = ((2 - gray)**2 - (2 - red)**2)/4
# Beta needs to be small enough that pixel with value 9 in the right part is red
range_end = ((9 - red)**2 - (9 - gray)**2)/4
print(f'{range_start, range_end}')

#%% 
# QUESTION 11, Energy evolution
''' The external energy needs to be decreasing (this is because we start from a 
bad separation into bright and dark and we end with a good separation). The
internal energy needs to be increasing (this is because we start with a small 
circle and we end with larger convoluted shape.) This leaves us with options (a)
and (d). But the overall energy needs to be decreasing, so we choose (d) as 
correct answer.
'''

#%% 
# QUESTION 12, Curve deformation
''' 
We consider expression for fext, in particular the part (I - 0.5(mean_in + mean_out)), 
since we know that the first part of fext positive. 

The points of the snake which are no longer at the boundary have large 
intensities I, so expression will be positive and those points 
will move outwards. 

The points which are at the boundary have intensities close to 
0.5(mean_amoeba + mean_background). However, the curve is not aligned with the
boundary of the amoeba. So while mean_in is equal to mean_amoeba, the value of
mean_out is larger than mean_background. Therefore, the value of 
(I - 0.5(mean_in + mean_out)) will be negative and points on the boundary of the 
amoeba will move (slightly) inwards.

(As snake aligns with the boundary of the amoeba, the mean_in and mean_out will
converge to the mean values of amoeba and background.)

 Correct answer is (f).
'''


#%% 
# QUESTION 13, Segmentation cost

# When interpreting the segmented image, if in doubt about which chanel was used 
# for above and which for below, notice that the segmented image has been obtained
# using this choice of costs function. So red channel is used for the red part of 
# the image and green channel for the green part of the image. Had the channels 
# been swapped, the segmentation would be different. (Explanation of the term
# 'below' and 'above' comes from the fact that image origin with height 0 is in 
# the top left corner.) 
I = skimage.io.imread(data_dir + 'fabric/fabric.png')
S = skimage.io.imread(data_dir + 'fabric/fabric_segmentation.png') > 0
pixelwise_cost = I[..., 0] * (~S) + I[..., 1] * S
fig, ax = plt.subplots(1, 3)
ax[0].imshow(I)
ax[1].imshow(S)
ax[2].imshow(pixelwise_cost)
ax[2].set_title(pixelwise_cost.sum())
plt.show()


#%% 
# QUESTION 14, Two-line detection

I = np.loadtxt(data_dir + 'image/image.txt').astype(int)
s = I.sum(axis=1)
c = s[3:] + s[:-3]
plt.imshow(I)
for i in range(len(s)):
    plt.text(I.shape[1] + 0.5, i, s[i])
for i in range(len(c)):
    plt.text(I.shape[1] + 1.5, i, c[i])

k = np.argmin(c)

plt.title(f'{min(c)} = {s[k]} + {s[k+3]}')
plt.show()


#%% 
# QUESTION 15, Classification output

X = np.array([[-0.1, 0.5], [0.1, 0.8], [0.2, 0], [0.9, -0.1]])
answer = []

for x in X:
    x2 = x[1]
    x1 = x[0]
    h2 = max(0.1 * x2 + 3 * x1 - 2.4, 0.0)
    h1 = max(1.5 * x2 + 0.4 * x1 + 1.2, 0.0)
    y2 = -2.2 * h2 + 1.8 * h1 - 3.2
    y1 = 0.5 * h2 - 2.7 * h1 + 2.7
    print()
    print(f'   x = {x1:0.2}, {x2:0.2} ')
    print(f'   h = {h1:0.2}, {h2:0.2} ')
    print(f'yhat = {y1:0.2}, {y2:0.2} ')
    print(f'   class {(y2>y1) + 1}')
    answer.append((y2>y1) + 1)

print()
print(answer)


#%% 
# QUESTION 16, MLP backpropagation
'''
For computing gradients in the backward pass you need activated values so (a) is 
not correct. The derivative of activation may be obtained from the activated values
so (c) is not correct -- as explained on the page 72 of the course note. As for 
(d), weights are updated when for gradients computed from current values, so it 
is not correct. The correct answer is (b) 

'''


#%% 
# QUESTION 17, Gradients
''' Every y_i is only depending on yhat_i. Therefore, the chain rule gives:
 delta_i = partial L / partial y_i  * partial y_i / partial yhat_i. The
 first part is derivative of quadratic function, so 2 * (y_i - t_i). The second 
 part is derivative or ReLU, which is either 0 or 1, depending on whether the
 neuron was activated.
'''

yhat = np.array([0.1 , -0.1, 0.4, 0.7, -0.2])
y = np.maximum(yhat, 0)
t = np.array([0 , 0, 1, 0, 0])
delta = 2 * (y - t) * (y > 0)
print(delta)

#%%
# QUESTION 18
'''Minibatches are not related to adaptive learning rate, memory considerations, 
inferrence. Minibatches are used in training to obtain averaged (smoothed) gradients. 
Correct anwer is (b)'''


#%%
# QUESTION 19, Network output

dims = np.array([114, 102])
down1 = (dims - 2)//2
down2 = (down1 - 2)//2
bottom = (down2 - 2)
up1 = 2 * bottom - 2
up2 = 2 * up1 - 2
print(up2)

# #%% ...IF SOMEBODY DECIDES TO USE TORCH
# import torch
# class exam_model(torch.nn.Module):
    
#     def __init__(self, out_channels=2):
#         super().__init__()

#         # learnable
#         self.conv1 = torch.nn.Conv2d(3, 8, 3)  
#         self.conv2 = torch.nn.Conv2d(8, 16, 3)  
#         self.conv3 = torch.nn.Conv2d(16, 32, 3)  
#         self.conv4 = torch.nn.Conv2d(32, 16, 3)  
#         self.conv5 = torch.nn.Conv2d(16, 8, 3)  
#         self.convfinal = torch.nn.Conv2d(8, 1, 1)         
#         self.convtrans34 = torch.nn.ConvTranspose2d(32, 16, 2, stride=2) 
#         self.convtrans45 = torch.nn.ConvTranspose2d(16, 8, 2, stride=2)
        
#         # convenience
#         self.relu = torch.nn.ReLU()
#         self.pool = torch.nn.MaxPool2d(2, 2)        
       
#     def forward(self, x):
 
#         # down
#         l1 = self.relu(self.conv1(x))
#         l2 = self.relu(self.conv2(self.pool(l1)))
#         out = self.relu(self.conv3(self.pool(l2))) 
        
#         # up         
#         out = torch.cat([self.convtrans34(out), l2[:,:,2:-2,2:-2]], dim=1)
#         out = self.relu(self.conv4(out))
#         out = torch.cat([self.convtrans45(out), l1[:,:,8:-8,8:-8]], dim=1)      
#         out = self.relu(self.conv5(out))
   
#          # finishing
#         out = self.convfinal(out)
  
#         return out
    
# model = exam_model()
# image_in = torch.randn(1, 3, 114, 102)
# image_out = model(image_in)
# print(image_in.shape)
# print(image_out.shape)


#%%
# QUESTION 20
'''Network takes RGB images so there are 3 input channels and therefore
the parameters of convolution are 3 * 3 * 3 * 8 + 8, the last 8 being biases. 
Max pooling and ReLU have no learnable parameters. So correct answer is 224.'''

# #%% ...IF SOMEBODY DECIDES TO USE TORCH
# import torch
# class block(torch.nn.Module):
    
#     def __init__(self):
#         super().__init__()
#         self.conv= torch.nn.Conv2d(3, 8, 3)  
#         self.relu = torch.nn.ReLU()
#         self.pool = torch.nn.MaxPool2d(2, 2)        
       
#     def forward(self, x):
#         return self.pool(self.relu(self.conv(x)))
            
# sum = 0 
# for p in block().parameters():
#     sum += p.numel()
# print(sum)


# %%
