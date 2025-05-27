import numpy as np 
import matplotlib.pyplot as plt 


def transform_pt(p, s, R, t):
    """
    Applies an affine transformation to a 2D point using the formula q = s * R * p + t.
    
    This function performs the following steps:
    1. Converts the input point p (provided as a tuple (x, y)) into a 2x1 column vector.
    2. Converts the translation vector t (provided as a tuple (tx, ty)) into a 2x1 column vector.
    3. Computes the matrix multiplication of the rotation matrix R with the point vector, then scales the result by s.
    4. Adds the translation vector to the scaled, rotated point.
    5. Flattens the resulting column vector back into a tuple (x', y').
    
    Parameters:
    p : tuple
        The input point as (x, y).
    s : float
        The uniform scaling factor.
    R : numpy.ndarray
        The 2x2 rotation matrix.
    t : tuple
        The translation vector as (tx, ty).
        
    Returns:
    tuple
        The transformed point as (x', y').
    """
    # Convert the input point to a numpy array and reshape it into a 2x1 column vector.
    p = np.array(p).reshape(2, 1)
    
    # Convert the translation vector to a numpy array and reshape it into a 2x1 column vector.
    t = np.array(t).reshape(2, 1)
    
    # Compute the transformation: scale and rotate the point, then translate it.
    q = s * np.dot(R, p) + t
    
    # Convert the resulting 2x1 column vector back into a flat tuple.
    return tuple(q.flatten())

def get_rotation_matrix(theta):
    """
    Returns a 2D rotation matrix corresponding to a counterclockwise rotation by the angle theta (in radians).
    
    The rotation matrix is defined as:
        [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]]
         
    Parameters:
    theta : float
        The rotation angle in radians.
    
    Returns:
    numpy.ndarray
        A 2x2 numpy array representing the rotation matrix.
    """
    # Calculate the cosine and sine of the angle theta.
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Construct and return the 2x2 rotation matrix.
    return np.array([[cos_theta, -sin_theta],
                     [sin_theta,  cos_theta]])


def compute_scale(p_set, q_set):
    """
    Computes the uniform scale factor between two sets of 2D points.
    
    The scale factor is determined by calculating the ratio between the sum of the distances 
    of each point in q_set from its centroid and the sum of the distances of each point in 
    p_set from its centroid. This effectively estimates how much p_set needs to be scaled 
    to match q_set.
    
    Parameters:
    p_set : list of tuples or list of lists
        The original set of 2D points. Each point is represented as (x, y).
    q_set : list of tuples or list of lists
        The transformed set of 2D points corresponding to p_set. Each point is represented as (x, y).
    
    Returns:
    float
        The computed scale factor, s, such that the distances in q_set are s times those in p_set.
    """
    # Convert input sets of points to numpy arrays for efficient computation.
    p_set, q_set = np.array(p_set), np.array(q_set)
    
    # Compute the centroids (mean values) of p_set and q_set along x and y axes.
    mu_p = np.mean(p_set, axis=0)
    mu_q = np.mean(q_set, axis=0)
    
    # Calculate the total Euclidean distance from each point in p_set to its centroid (mu_p).
    dist_p = np.sum([np.linalg.norm(p - mu_p) for p in p_set])
    
    # Calculate the total Euclidean distance from each point in q_set to its centroid (mu_q).
    dist_q = np.sum([np.linalg.norm(q - mu_q) for q in q_set])
    
    # Compute the scale factor as the ratio of total distances in q_set to that in p_set.
    s = dist_q / dist_p
    
    return s


def covariance_matrix(p_set, q_set):
    # Convert input point sets from list to numpy arrays for efficient vectorized operations.
    p_set, q_set = np.array(p_set), np.array(q_set)

    # Compute the centroids (mean of x and y coordinates) for p_set and q_set.
    mu_p = np.mean(p_set, axis=0)
    print("The mean of p is ", mu_p.shape)  # Debug print: shows the shape of the centroid of p_set.
    mu_q = np.mean(q_set, axis=0)

    # Center the data by subtracting the centroid from each point.
    # This shifts the origin to the centroid of the point sets.
    p_centered = p_set - mu_p
    q_centered = q_set - mu_q

    # Compute the cross-covariance matrix C using the centered points.
    # This matrix captures the correlation between the point sets.
    C = np.dot(q_centered.T, p_centered)
    
    # Perform Singular Value Decomposition on the covariance matrix.
    # This decomposes C into U * sigma * Vt, where:
    # - U and Vt are orthogonal matrices representing rotations,
    # - sigma contains the singular values representing scaling factors.
    U, sigma, Vt = np.linalg.svd(C)

    # Calculate the initial estimated rotation matrix R_hat.
    # Multiplying U and Vt gives a rotation that aligns p_set to q_set.
    R_hat = U @ Vt

    # Compute the determinant of R_hat to check if it represents a proper rotation.
    # A negative determinant indicates a reflection, not a proper rotation.
    det = np.linalg.det(R_hat)

    # Handle the reflection case: if det is negative, adjust R_hat.
    if det < 0:
        # Create a diagonal matrix D to correct the reflection.
        # The second diagonal element is set to the determinant to invert the reflection.
        D = np.array([[1, 0], 
                      [0, det]])
        # Modify R_hat by multiplying with D to ensure a proper rotation.
        R_hat = R_hat @ D

    # Debug prints: output shapes of U, sigma, Vt and the sign of the determinant.
    print("The shape of U is ", U.shape)
    print("The shape of sigma is ", sigma.shape)
    print("The shape of Vt is ", Vt.shape)
    print("Sign of det is ", det)
    print(U)

    # Return the final rotation matrix which aligns the point sets.
    return R_hat

def translation(p_set, q_set, R, s):
    # Convert input sets to NumPy arrays for vectorized operations.
    # This ensures that calculations can be performed efficiently.
    p_set, q_set = np.array(p_set), np.array(q_set)

    # Compute the centroid (mean point) of p_set.
    # The centroid is the average of all points in p_set.
    mu_p = np.mean(p_set, axis=0)

    # Compute the centroid (mean point) of q_set.
    # The centroid is the average of all points in q_set.
    mu_q = np.mean(q_set, axis=0)

    # Calculate the translation vector t.
    # Here, we aim to find t such that when we apply scaling s and rotation R to mu_p,
    # then add the translation vector t, we obtain mu_q.
    # The equation used is: t = mu_q - s * (R * mu_p)
    t = mu_q - s * np.dot(R, mu_p)

    # Flatten the translation vector to ensure it's a 1D tuple with two elements,
    # corresponding to the x and y translation components.
    # Returning a tuple makes the output consistent with the rest of the code.
    return t 
