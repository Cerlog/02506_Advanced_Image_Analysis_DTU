# Exercise 1.1.1 Image convolution

- Read the image using [`skimage.io.imread`](https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread)
- For convolution use [`scipy.ndimage.convolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html)
- After completing the exercise test agains [`scipy.ndimage.gaussian_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)
  

# Exercise 1.1.3 Curve smoothing

- Make the matrix using [`scipy.linalg.circulant`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.circulant.html)
- For matrix-vector multiplication you can use [`numpy.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) or the shorthand operator `@` (explained at the bottom of page dedicated to `matmul`).
  
# Exericse 1.1.6 Working with volumetric image
- To get hold of all image in a certain folder, use `sorted(os.listdir(\<FOLDER NAME\>))`

# Exercise 1.1.7 PCA of multispectral image
- For eigendecomposition use `numpy.linalg.eig`
- You can compare your PCA against the existing implementation `sklearn.decomposition.PCA`

# Exercise 1.1.8 Bacterial growth from movie frames
- To read the movie as a list of images, use following code block
```python
import imageio
import skimage

filename = ...  # filename of the mp4 movie
vid = imageio.get_reader(filename,  'ffmpeg')

frames = []
for v in vid.iter_data():
    frames.append(skimage.color.rgb2gray(v))
```