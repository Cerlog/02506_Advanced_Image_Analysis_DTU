import numpy as np
import scipy.interpolate
import scipy.linalg
import skimage.draw
import matplotlib.pyplot as plt


def calculate_internal_energy(snake, alpha=0.01, beta=0.1):
    """Calculate internal energy components at each snake point.
    
    Returns:
        continuity_energy: Energy due to stretching (first derivative)
        curvature_energy: Energy due to bending (second derivative)
        total_internal: Combined internal energy
    """
    N = len(snake)
    
    # Calculate first derivatives (stretching energy)
    d1 = np.roll(snake, -1, axis=0) - snake
    continuity_energy = alpha * np.sum(d1**2, axis=1)
    
    # Calculate second derivatives (curvature energy)
    d2 = np.roll(snake, -1, axis=0) - 2*snake + np.roll(snake, 1, axis=0)
    curvature_energy = beta * np.sum(d2**2, axis=1)
    
    total_internal = continuity_energy + curvature_energy
    
    return continuity_energy, curvature_energy, total_internal


def calculate_external_energy(snake, I):
    """Calculate external energy at each snake point."""
    mask = skimage.draw.polygon2mask(I.shape, snake)
    m_in = np.mean(I[mask])
    m_out = np.mean(I[~mask])
    
    # Interpolate image values at snake points
    f = scipy.interpolate.RectBivariateSpline(
        np.arange(I.shape[0]), np.arange(I.shape[1]), I
    )
    val = f(snake[:, 0], snake[:, 1], grid=False)
    
    # External energy (region-based)
    f_ext = (m_in - m_out) * (2 * val - m_in - m_out)
    
    return f_ext


def plot_snake_energies(snake, I, alpha=0.01, beta=0.1, save_plots=True):
    """Plot internal and external energies as curves along the snake contour."""
    
    # Calculate energies
    continuity_energy, curvature_energy, total_internal = calculate_internal_energy(snake, alpha, beta)
    external_energy = calculate_external_energy(snake, I)
    
    # Create parameter along snake (normalized arc length)
    N = len(snake)
    t = np.arange(N)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: External Energy
    axes[0, 0].plot(t, external_energy, 'b-', linewidth=2, label='External Energy')
    axes[0, 0].set_xlabel('Snake Point Index')
    axes[0, 0].set_ylabel('External Energy')
    axes[0, 0].set_title('External Energy Along Snake Contour')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Internal Energy Components
    axes[0, 1].plot(t, continuity_energy, 'r-', linewidth=2, label=f'Continuity (α={alpha})')
    axes[0, 1].plot(t, curvature_energy, 'g-', linewidth=2, label=f'Curvature (β={beta})')
    axes[0, 1].plot(t, total_internal, 'k--', linewidth=2, label='Total Internal')
    axes[0, 1].set_xlabel('Snake Point Index')
    axes[0, 1].set_ylabel('Internal Energy')
    axes[0, 1].set_title('Internal Energy Components Along Snake Contour')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Total Energy
    total_energy = external_energy + total_internal
    axes[1, 0].plot(t, total_energy, 'm-', linewidth=2, label='Total Energy')
    axes[1, 0].plot(t, external_energy, 'b--', alpha=0.7, label='External')
    axes[1, 0].plot(t, total_internal, 'r--', alpha=0.7, label='Internal')
    axes[1, 0].set_xlabel('Snake Point Index')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Total Energy Along Snake Contour')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Snake visualization with energy color coding
    axes[1, 1].imshow(I, cmap='gray')
    scatter = axes[1, 1].scatter(snake[:, 1], snake[:, 0], 
                                c=total_energy, cmap='viridis', s=20)
    axes[1, 1].set_title('Snake with Total Energy Color Coding')
    plt.colorbar(scatter, ax=axes[1, 1], label='Total Energy')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('snake_energy_analysis.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return external_energy, continuity_energy, curvature_energy, total_internal


def evolve_snake_with_energy_plots(snake, I, B, step_size, alpha=0.01, beta=0.1):
    """Modified snake evolution function that plots energy curves."""
    
    # Calculate external energy
    external_energy = calculate_external_energy(snake, I)
    
    # Calculate displacement
    displacement = step_size * external_energy[:, None] * get_normals(snake)
    
    # Apply external forces
    snake_after_external = snake + displacement
    
    # Apply internal forces (smoothing)
    snake_after_internal = B @ snake_after_external
    
    # Plot energies before evolution
    print("Energy analysis before evolution step:")
    plot_snake_energies(snake, I, alpha, beta)
    
    # Continue with standard evolution
    snake_final = remove_intersections(snake_after_internal)
    snake_final = distribute_points(snake_final)
    snake_final = keep_snake_inside(snake_final, I.shape)
    
    return snake_final


# Additional utility functions from your original code
def make_circular_snake(N, center, radius):
    """ Initialize circular snake."""
    center = np.asarray(center).reshape([1, 2])
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    unit_circle = np.array([np.cos(angles), np.sin(angles)]).T
    return center + radius * unit_circle


def normalize(n):
    l = np.sqrt((n ** 2).sum(axis=1, keepdims = True))
    l[l == 0] = 1
    return n / l


def get_normals(snake):
    """ Returns snake normals. """
    ds = normalize(np.roll(snake, 1, axis=0) - snake) 
    tangent = normalize(np.roll(ds, -1, axis=0) + ds)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return normal 


def distribute_points(snake):
    """ Distributes snake points equidistantly."""
    N = len(snake)
    closed = snake[np.hstack([np.arange(N), 0])]
    d = np.sqrt(((np.roll(closed, 1, axis=0) - closed) ** 2).sum(axis=1))
    d = np.cumsum(d)
    d = d / d[-1]  # Normalize to 0-1
    x = np.linspace(0, 1, N, endpoint=False)  # New points
    new =  np.stack([np.interp(x, d, closed[:, i]) for i in range(2)], axis=1) 
    return new


def is_ccw(A, B, C):
    # Check if A, B, C are in counterclockwise order
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def is_crossing(A, B, C, D):
    # Check if line segments AB and CD intersect
    return is_ccw(A, C, D) != is_ccw(B, C, D) and is_ccw(A, B, C) != is_ccw(A, B, D)


def is_counterclockwise(snake):
    """ Check if points are ordered counterclockwise."""
    return np.dot(snake[1:, 0] - snake[:-1, 0],
                  snake[1:, 1] + snake[:-1, 1]) < 0


def remove_intersections(snake, method = 'new'):
    """ Reorder snake points to remove self-intersections."""
    N = len(snake)
    closed = snake[np.hstack([np.arange(N), 0])]
    for i in range(N - 2):
        for j in range(i + 2, N):
            if is_crossing(closed[i], closed[i + 1], closed[j], closed[j + 1]):
                # Reverse vertices of smallest loop
                rb, re = (i + 1, j) if j - i < N // 2 else (j + 1, i + N)
                indices = np.arange(rb, re+1) % N                 
                closed[indices] = closed[indices[::-1]]                              
    snake = closed[:-1]
    return snake if is_counterclockwise(snake) else np.flip(snake, axis=0)


def keep_snake_inside(snake, shape):
    """ Contains snake inside the image."""
    snake[:, 0] = np.clip(snake[:, 0], 0, shape[0] - 1)
    snake[:, 1] = np.clip(snake[:, 1], 0, shape[1] - 1)
    return snake

    
def regularization_matrix(N, alpha, beta):
    """ Matrix for smoothing the snake."""
    s = np.zeros(N)
    s[[-2, -1, 0, 1, 2]] = (alpha * np.array([0, 1, -2, 1, 0]) + 
                    beta * np.array([-1, 4, -6, 4, -1]))
    S = scipy.linalg.circulant(s)  
    return scipy.linalg.inv(np.eye(N) - S)


# Example usage:
if __name__ == "__main__":
    # Create a simple test image and snake
    I = np.zeros((100, 100))
    I[30:70, 30:70] = 1  # Simple square object
    
    # Initialize circular snake
    snake = make_circular_snake(50, [50, 50], 20)
    
    # Create regularization matrix
    N = len(snake)
    alpha, beta = 0.01, 0.1
    B = regularization_matrix(N, alpha, beta)
    
    # Plot energy analysis
    plot_snake_energies(snake, I, alpha, beta)