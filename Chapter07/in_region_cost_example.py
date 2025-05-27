import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import slgbuilder

# Fix deprecated numpy aliases
np.bool = bool
np.int = int

# Load the image
RGB = skimage.io.imread('rammed-earth-layers-limestone.jpg').astype(np.int32)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(RGB)
I = np.mean(RGB, axis=2)
ax[0,0].set_title('Input')

#%% SETTINGS FOR GEOMETRIC CONSTRAINS
delta = 1 # smoothness very constrained, try also 3 to see less smoothness

#%% DARKEST LINE
layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)

helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1

ax[0,1].imshow(RGB)
ax[0,1].plot(segmentation_line, 'r')
ax[0,1].set_title('Darkest line')

#%% TWO DARK LINES
layers = [slgbuilder.GraphObject(I),slgbuilder.GraphObject(I)]
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)
helper.add_layered_containment(layers[0], layers[1], min_margin=50, max_margin=200)

helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

ax[1,0].imshow(RGB)
for line in segmentation_lines:
    ax[1,0].plot(line, 'r')
ax[1,0].set_title('Two dark lines')

#%% DARKEST REGION
layers = [slgbuilder.GraphObject(0*I), slgbuilder.GraphObject(0*I)] # no on-surface cost
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)
helper.add_layered_boundary_cost()
helper.add_layered_region_cost(layers[0], 255-I, I)
helper.add_layered_region_cost(layers[1], I, 255-I)
helper.add_layered_smoothness(delta=delta, wrap=False)  
helper.add_layered_containment(layers[0], layers[1], min_margin=1, max_margin=200)

helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

ax[1,1].imshow(RGB)
for line in segmentation_lines:
    ax[1,1].plot(line, 'r')
ax[1,1].set_title('Dark region')
plt.tight_layout()
plt.show()