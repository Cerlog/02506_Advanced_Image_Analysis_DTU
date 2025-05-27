import skimage.io
import numpy as np
import matplotlib.pyplot as plt

#%% Question 1 - Gaussian derivative
# Answer (b)
s = 1
x = np.arange(-5*s,5*s+1)[:,None]
g = 1 / (s*np.sqrt(2*np.pi)) * np.exp(-x**2/(2*s**2))
dg = -x / s**2*g
kernel = g @ dg.T

fig, ax = plt.subplots(1,2)
ax[0].plot(x,g)
ax[0].plot(x,dg)
ax[1].imshow(kernel)

print('Question 1: b')
print(np.round(kernel[3:-3, 3:-3],2))


#%% Question 2 - Scale-space blob detection
# answer (e)
import skimage.feature

L = skimage.io.imread('../data/blobs_Laplace.tif')
coords = skimage.feature.peak_local_max(-L, threshold_abs=60)
sc, r, c = coords[0]
scale = 16*(sc+1)
print('Question 2:') 
print(f'  Scale: {scale}, position: ({r},{c})')

# From here just checking
fig, ax = plt.subplots()
im = skimage.io.imread('../data/blobs.png')
ax.imshow(im, cmap='gray')
theta = np.linspace(0, 2*np.pi, 200)
d = np.sqrt(2*scale)
x = np.cos(theta)*d + c
y = np.sin(theta)*d + r
ax.plot(x, y, 'r')


#%% Question 3 - Feature-based segmentation
# Answer (c)
label1 = np.array([4, 350, 1])
label2 = np.array([45, 147, 36])

label = label1 + label2
prob = label/label.sum()
print(f'Question 3: {np.round(prob,3)}')


#%% Question 4 - Segmentation with image patches
# Answer (c)
# Kernel 7 removes 6 gives 506, 4 channels for 4 labels
print('Question 4: 506x506x4')


#%% Question 5 - Matching SIFT features
# Answer (b)
desc1 = np.loadtxt('../data/SIFT_1_descriptors.txt')
desc2 = np.loadtxt('../data/SIFT_2_descriptors.txt')

coord1 = np.loadtxt('../data/SIFT_1_coordinates.txt')
coord2 = np.loadtxt('../data/SIFT_2_coordinates.txt')

d = ((desc1[0:1]-desc2)**2).sum(axis=1)
idx = np.argmin(d)
dist = np.sqrt(((coord1[0] - coord2[idx])**2).sum())
print(f'Question 5: {dist}')


#%% Question 6 - Feature-based image registration
# Answer (f)
C = np.array([[191955, -937044],[552183, 379358]])

U, S, V = np.linalg.svd(C)
R = U @ V
print(f'Question 6: {np.arccos(R[0,0])/np.pi*180}')

#%% Question 7 - Gaussian Convolution
# Answer (b)
print('Question 7: b')


#%% Question 8 - MRF prior energy
# Answer (j)
I = np.array([[1, 1, 1, 1, 2, 2, 2],
              [1, 2, 2, 1, 2, 3, 2],
              [2, 3, 2, 1, 2, 3, 2],
              [2, 2, 2, 1, 1, 3, 2],
              [1, 2, 1, 1, 1, 1, 1,]]) 
U2 = 10*(np.abs(np.diff(I, axis=0)).sum() + np.abs(np.diff(I, axis=1)).sum())
print(f'Question 8: {U2}')


#%% Question 9 - Likelihood energy
# Answer (g)
import scipy.ndimage

def window_std(I):
    window = np.ones(shape=(5, 5))/25
    K_I2 = scipy.ndimage.convolve(I**2, window, mode='reflect')
    KI_2 = scipy.ndimage.convolve(I, window, mode='reflect')**2
    return np.sqrt(K_I2 - KI_2)

I = skimage.io.imread('../data/blended.png')/255
S = window_std(I)

sigma_1 = 0.05  
sigma_2 = 0.01  

U1 = (sigma_1 - S)**2
U2 = (sigma_2 - S)**2

B = U1 > U2  # max likelihood segmentation
L = (1 - B) * U1 + B * U2   # one-clique potentials for this segmentation

fig, ax = plt.subplots(1, 2)
ax[0].imshow(B)
ax[0].set_title('max likelihood')
ax[1].imshow(L)
ax[1].set_title(f'U: {L.sum()}')
print(f'Question 9: {L.sum()}')


#%% Question 10 - Exterrnal forces
# Answer (d)
m_in = 0
m_out = 90*90/(100*200-90*90)
f = (m_in-m_out)*(0-m_in-m_out)
print(f'Question 10: {f}')


#%% 11 - Curve smoothing
# Answer (a)

# taken from weekly solutions at gitlab 
def regularization_matrix(N, alpha, beta):
    """An NxN matrix for imposing elasticity and rigidity to snakes.
    Arguments: alpha is weigth for second derivative (elasticity),
    beta is weigth for (-)fourth derivative (rigidity)."""
    d = alpha*np.array([-2, 1, 0, 0]) + beta*np.array([-6, 4, -1, 0])
    D = np.fromfunction(lambda i,j: np.minimum((i-j)%N,(j-i)%N), (N,N), dtype=int)
    A = d[np.minimum(D,len(d)-1)]
    return(np.linalg.inv(np.eye(N)-A))

# taken from weekly solutions at gitlab
def curve_length(X):
    d = X - np.roll(X, shift=1, axis=0)
    d = (d**2).sum(axis=1)
    d = (np.sqrt(d)).sum()
    return(d)
    
X = np.loadtxt('../data/hand_noisy.txt')

fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1])
ax.set_aspect('equal')

X_alpha = regularization_matrix(X.shape[0], 100, 0) @ X
X_beta = regularization_matrix(X.shape[0], 0, 100) @ X

ax.plot(X_alpha[:, 0], X_alpha[:, 1], 'r')
ax.plot(X_beta[:, 0], X_beta[:, 1], 'g')

d = curve_length(X_alpha) - curve_length(X_beta)
print(f'Question 11: {d}')


#%% Question 12 - Two surface cost
# Answer (c)
s1 = np.array([80, 93, 91, 82, 92, 86, 87, 88, 85, 82])
s2 = np.array([24, 25, 39, 32, 21, 21, 32, 21, 39, 30])

c = ((s1-90)**2).sum() + ((s2-30)**2).sum()
print(f'Question 12: {c}')


#%% Question 13 Surface distance
# Answer (d)
C = np.loadtxt('../data/cost.txt', dtype=int)
s0 = C.sum(axis=1)
s0 = np.argmin(s0)
s5 = np.argmin(C, axis=0)
d = (abs(s0-s5))
print(f'Question 13: {d.mean()}')


#%% Question 14 Classification loss
# Answer (e)
in_t_out = np.loadtxt('../data/in_t_out.txt')
x = in_t_out[:, 0:3]
t = in_t_out[:, 3].astype(int)
yhat = in_t_out[:, 4:]

y = np.exp(yhat)
y = y/y.sum(axis=1, keepdims=True)

loss = -np.log(y[range(len(y)), t]).sum()
print(f'Question 14: {loss}')

#%% Question 15 - MLP classification
# Answer (d)
W1 = np.loadtxt('../data/W1.txt')
W2 = np.loadtxt('../data/W2.txt')
W3 = np.loadtxt('../data/W3.txt')

def mlp(x):
    h1 = np.maximum(np.concatenate(([1], x)) @ W1, 0)
    h2 = np.maximum(np.concatenate(([1], h1)) @ W2, 0)
    y_hat = np.concatenate(([1], h2)) @ W3
    y = np.exp(y_hat)
    y = y/y.sum()
    return np.argmax(y)
    
colors = 'rgbc'
fig, ax = plt.subplots()
for i in range(1000):
    x = np.random.uniform(low=-1, high=1, size=2)
    y = mlp(x)
    ax.plot(x[0], x[1], '.'+colors[y])
print('Question 15: d')

#%% Question 16 - MLP backpropagation
# Answer (d)
x = 0.5
h = [1, 0.8, -0.1+0.9*x, -0.1*x]
h = [max(n,0) for n in h]
yhat = [0.4*h[0], 0.5*h[0]+0.1*h[1]-0.2*h[2], 0.8*h[2]-0.2*h[3]]
y = [np.exp(n) for n in yhat]
y = [n/sum(y) for n in y]


dLdw = h[2]*(y[3-1]) # -1 because y starts with y1
print(f'Question 16: {dLdw}')

#%% Question 17 - Stochastic gradient decent
# Answer (b)
print('Question 17: b')

#%% Question 18 - CNN reseptive field
# Answer (j) 
# one out pixel -> 1 
# max pool doubles -> 2 (receptive field before max pool)
# 3-by-3 convolution adds 2 -> 4
# max pool doubles -> 8 
# 7-by-7 convolution adds 6 -> 14
r = (1*2 + 2)*2 + 6
print(f'Question 18: {r}x{r}')
#%% Question 19 - CNN parameters
# Answer (h) 
n_param = (1 + 7*7*3)*8 + (1 + 3*3*8)*16
print(f'Question 19: {n_param}')

#%% Question 20 - Data augmentation
# Answer (a)
print('Question 20: a')