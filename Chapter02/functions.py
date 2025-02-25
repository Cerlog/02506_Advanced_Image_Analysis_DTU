from scipy.ndimage import convolve
from scipy.linalg import circulant
import numpy as np
import skimage
from loguru import logger
import matplotlib.pyplot as plt


def smooth_image_2D(image, kernel_2d):
    """
    Smooth an image using a 2D kernel and calculate difference
    
    Args:
        image (ndarray): Input image
        kernel_2d (ndarray): 2D convolution kernel
        
    Returns:
        tuple: (smoothed image, difference image)
    """
    # apply convolution
    smoothed = convolve(image, kernel_2d, mode='reflect')
    
    # calculate difference
    difference = image - smoothed
    
    return smoothed, difference


def create_1D_kernel(sigma, x):
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def derivative_1D_kernel(sigma, x):
    kernel = -x/(sigma**2) * np.exp(-0.5 * (x/sigma)**2)
    return kernel


def create_1D_kernel_2(t, n, return_x = False):
    sigma = np.sqrt(t)
    print(f"Sigma: {sigma}")
    radius = int(np.ceil(sigma * n))
    x = np.arange(-radius, radius + 1) 
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    if  return_x == True: 
        return kernel, x
    return kernel 


def create_1D_kernel_grad_2(t, n, return_x = False):
    sigma = np.sqrt(t)
    print(f"Sigma: {sigma}")
    radius = int(np.ceil(sigma * n))
    x = np.arange(-radius, radius + 1) 
    kernel = -x/(sigma**2) * np.exp(-0.5 * (x/sigma)**2)
    if return_x == True:
        return kernel, x
    return kernel



def apply_separable_gaussian_convolution(image, kernel_1D):
    """
    Apply separable Gaussian convolution to an image using 1D kernels.
    
    Args:
        image (numpy.ndarray): Input image
        kernel_1D (numpy.ndarray): 1D Gaussian kernel
        
    Returns:
        numpy.ndarray: Blurred image after applying separable convolution
        numpy.ndarray: Difference between original and blurred image
    """
    # Convolve column wise
    column_wise_gaussian_kernel = kernel_1D.reshape(-1, 1)
    image_columns = convolve(image, column_wise_gaussian_kernel)
    
    # Convolve along the rows
    row_wise_gaussian_kernel = kernel_1D.reshape(1, -1)
    image_rows = convolve(image_columns, row_wise_gaussian_kernel)
    
    # Calculate difference between original and blurred image
    difference = image - image_rows
    
    return image_rows, difference


def convolve_columns(image, kernel_1D):
    """
    Apply 1D convolution along columns.
    
    Args:
        image (numpy.ndarray): Input image.
        kernel_1D (numpy.ndarray): 1D Gaussian kernel.
    
    Returns:
        numpy.ndarray: Image after column-wise convolution.
    """
    column_wise_gaussian_kernel = kernel_1D.reshape(-1, 1)
    return convolve(image, column_wise_gaussian_kernel)

def convolve_rows(image, kernel_1D):
    """
    Apply 1D convolution along rows.
    
    Args:
        image (numpy.ndarray): Input image.
        kernel_1D (numpy.ndarray): 1D Gaussian kernel.
    
    Returns:
        numpy.ndarray: Image after row-wise convolution.
    """
    row_wise_gaussian_kernel = kernel_1D.reshape(1, -1)
    return convolve(image, row_wise_gaussian_kernel)


def get_gray_image(path):
    image_original = skimage.io.imread(path)
    return skimage.img_as_float(image_original)



def length_of_segmentation_boundary(image):
    # create a coppy of the image 
    image_copy = image.copy()
    logger.info(f"Original image values: {np.unique(image_copy)}")

    # determine the unique values in the image
    unique_values = np.unique(image_copy)
    logger.info(f"Unique values: {unique_values}")
    for i, value in enumerate(unique_values):
        logger.info(f"Mapping {value} to {i}")
        image_copy[image_copy == value] = i
    
    logger.info(f"After mapping, values: {np.unique(image_copy)}")

    # convert to int32 to avoid overflow of the unit8
    image_copy = image_copy.astype(np.int32)

    # calculate the row and column differences
    row_diff = np.abs(image_copy[1:] - image_copy[:-1])
    logger.info(f"Row differences: {np.unique(row_diff, return_counts=True)}")
    col_diff = np.abs(image_copy[:,:-1] - image_copy[:,1:])
    logger.info(f"Column differences: {np.unique(col_diff, return_counts=True)}")

    count_of_row_diff = np.unique(row_diff, return_counts=True)
    indexes = count_of_row_diff[0] > 0
    count_of_row_diff = sum(count_of_row_diff[1][indexes])
    logger.info(f"Count of row differences: {count_of_row_diff}")

    count_of_col_diff = np.unique(col_diff, return_counts=True)
    indexes = count_of_col_diff[0] > 0
    count_of_col_diff = sum(count_of_col_diff[1][indexes])
    logger.info(f"Count of column differences: {count_of_col_diff}")

    logger.info(f"Total length of segmentation boundary: {count_of_row_diff + count_of_col_diff}")
    

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(row_diff, cmap='gray')
    ax[0].set_title("Row differences")
    ax[1].imshow(col_diff, cmap='gray')
    ax[1].set_title("Column differences")
    plt.show()
    return count_of_row_diff + count_of_col_diff
    

def create_entry(**kwargs):
    """
    Creates a dictionary with the provided key-value pairs and adds a fixed entry for 'display_name'.
    
    Parameters:
        **kwargs: Arbitrary keyword arguments where the key is a string and the value is numerical.
        
    Returns:
        dict: A dictionary containing all provided key-value pairs plus {'display_name': 'pbn'}.
        
    Example:
        >>> entry = create_entry(image_convolution=42, boundary_length=10801, curve_smoothing=79)
        >>> print(entry)
        {'image_convolution': 42, 'boundary_length': 10801, 'curve_smoothing': 79, 'display_name': 'pbn'}
    """
    entry = kwargs.copy()
    entry["display_name"] = "pbn"
    return entry

def save_to_file(filename, data):
    """
    Saves the provided data dictionary into a text file in the specified format.
    
    Each key-value pair is written on a separate line in the format:
    
    key: value
    
    Parameters:
        filename (str): The name of the file to save the data into.
        data (dict): The dictionary containing data to be saved.
    """
    with open(filename, 'w') as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")





def smoothing_equation_1_10_2(data, LAMBDA=0.19, iterations=10, true_data=None):
    # Make a copy of the original data to avoid modifying it
    X_new = data.copy()
    
    # Ensure the curve is closed by making first and last points identical
    if not np.array_equal(X_new[0], X_new[-1]):
        X_new = np.vstack([X_new, X_new[0]])
    
    N = X_new.shape[0]
    # Construct the circulant matrix for cyclic boundary conditions
    L = np.zeros(N)
    L[0] = -2
    L[1] = 1
    L[-1] = 1
    L = circulant(L)

    I = np.eye(N)
    
    # Always apply the matrix multiplication once
    X_new = (I + LAMBDA * L) @ X_new
    

    # If iterations > 0, apply additional smoothing
    if iterations > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(True)
        ax.set_aspect('equal')
        for _ in range(iterations-1):  # -1 because we already did one iteration
            ax.clear()
            X_new = (I + LAMBDA * L) @ X_new
            # Plot the curve
            ax.plot(X_new[:,0], X_new[:,1], '-', color='m')
            # Add the closing line
            ax.plot([X_new[-1,0], X_new[0,0]], [X_new[-1,1], X_new[0,1]], '-', color='m')
            ax.set_title(rf"Smoothing with iterations={iterations}, $\lambda=${LAMBDA} (iteration)")
            ax.grid(True)
            ax.set_aspect('equal')
            clear_output(wait=True)
            display(fig)
            plt.pause(0.1)  # Add small pause to make animation visible

    plt.figure(figsize=(10, 8))
    # Plot the smoothed data
    plt.plot(X_new[:,0], X_new[:,1], '-', color='m')
    # Add a single line connecting end to start
    plt.plot([X_new[-1,0], X_new[0,0]], [X_new[-1,1], X_new[0,1]], '-', color='m')
    plt.title(rf"Smoothing with $\lambda=${LAMBDA}, iterations={iterations} (final result)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    if true_data is not None:
        plt.figure(figsize=(10, 8))
        plt.plot(true_data[:,0], true_data[:,1], '-', label='True Data')
        plt.plot(X_new[:,0], X_new[:,1], '-', label='Smoothed')
        # Add a single line connecting end to start for smoothed data
        plt.plot([X_new[-1,0], X_new[0,0]], [X_new[-1,1], X_new[0,1]], '-')
        plt.title(rf"Smoothing with $\lambda=${LAMBDA}, iterations={iterations} ")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    distance = calculate_distance_(X_new)
    print(f"The total distance of the curve is {distance:.4f}")
    print(f"Length of the dataset is {N}")
    return distance




def get_gaussian_kernels(s):
    '''
    Returns a 1D Gaussian kernel and its derivative and the x values.
    
    Parameters
    ----------
    s : float
        Standard deviation of the Gaussian kernel.
        
    Returns
    -------
    g : ndarray
        1D Gaussian kernel.
    dg : ndarray
        Derivative of the Gaussian kernel.
    x : ndarray
        x values where the Gaussian is computed.

    '''
    t = s**2
    r = np.ceil(4 * s)
    x = np.arange(-r, r + 1).reshape(-1, 1)
    g = np.exp(-x**2 / (2 * t))
    g = g/np.sum(g)
    dg = -x * g / t 
    return g, dg, x



def calculate_distance_(data1):
    # take the norm between x and y coordinates
    distances = np.linalg.norm(data1[1:] - data1[:-1], axis=1)
    # sum the distances
    distance = np.sum(distances)
    # save only 2 decimal places
    distance = np.round(distance, 2)
    return distance



def main():
    # Create the dictionary with the specified variables.
    data = create_entry(image_convolution=42, boundary_length=10801, curve_smoothing=79)
    
    name_of_file = "quiz3.txt"
    # Save the data into quiz.txt in the requested format.
    save_to_file(name_of_file, data)
    
    print(f"Data has been saved to {name_of_file}.")

if __name__ == "__main__":
    main()