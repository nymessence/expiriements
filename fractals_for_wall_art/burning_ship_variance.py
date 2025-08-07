import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from multiprocessing import Pool
from tqdm import tqdm

# Parameters
width, height = 8192, 8192
center_x, center_y = -.5, -.75 # re, im
scale = 3.5  # Reduced scale to zoom in and capture more detail
max_iter = 100
samples_per_radius = 10
radii = [1e-4, 1e-5, 1e-6]

# 23.592 MP/h on burning ship benchmark

@njit
def complex_cos(z):
    # cos(z) = cos(a + ib) = cos(a) * cosh(b) - i sin(a) * sinh(b)
    return np.cos(z.real) * np.cosh(z.imag) - 1j * np.sin(z.real) * np.sinh(z.imag)

@njit
def complex_pow(z, c):
    """
    Computes the power of a complex number, z^c, using the formula z^c = exp(c * log(z)).
    
    Args:
        z (complex): The base number.
        c (complex): The exponent.
        
    Returns:
        complex: The result of z raised to the power of c.
    """
    if z == 0:
        # Handle the special case where z is zero.
        # This avoids issues with taking the logarithm of zero.
        # The result is 0 if c has a positive real part, otherwise, it's typically undefined.
        # We'll return 0 for simplicity.
        return 0.0 + 0.0j

    # Calculate the complex logarithm of z.
    # log(z) = log(|z|) + i*arg(z)
    log_z_real = np.log(np.abs(z))
    log_z_imag = np.angle(z)
    log_z = log_z_real + 1j * log_z_imag
    
    # Compute the product c * log(z).
    product_real = c.real * log_z_real - c.imag * log_z_imag
    product_imag = c.real * log_z_imag + c.imag * log_z_real
    
    # Compute exp(c * log(z)).
    # exp(a + ib) = exp(a) * (cos(b) + i*sin(b))
    exp_real_part = np.exp(product_real)
    return exp_real_part * (np.cos(product_imag) + 1j * np.sin(product_imag))

# Example usage:
# z = 2 + 3j
# c = 1.5 + 2j
# result = complex_pow(z, c)
# print(result)


import numpy as np
from numba import njit

@njit
def mandelbrot_escape_speed(c_re, c_im, max_iter):
    c = complex(c_re, c_im)
    z = 0.0 + 0.0j  # Start with z = 0
    prev_log_mag = -100.0
    sum_log_diffs = 0.0
    count = 0

    for _ in range(max_iter):
        # This is the correct and robust way to compute z^2
        # uncomment to use function
        # z = z**2 + c # standard mandelbrot
        # z = z**z + c # exponential map
        z = complex(abs(z.real), abs(z.imag))**2 + c # burning ship
        
        mag = abs(z)
        if mag > 1e20:
            break
        
        log_mag = np.log2(mag + 1e-16)
        sum_log_diffs += (log_mag - prev_log_mag)
        prev_log_mag = log_mag
        count += 1

    return sum_log_diffs / count if count > 0 else 0.0

def burning_ship_escape_speed(c_re, c_im, max_iter):
    c = complex(c_re, c_im)
    z = 0.0 + 0.0j  # Start with z = 0
    prev_log_mag = -100.0
    sum_log_diffs = 0.0
    count = 0

    for _ in range(max_iter):
        # Burning Ship formula: z_new = (|Re(z)| + i|Im(z)|)^2 + c
        # The core change is here
        z = complex(abs(z.real), abs(z.imag))**2 + c
        
        mag = abs(z)
        if mag > 1e20:
            break
        
        log_mag = np.log2(mag + 1e-16)
        sum_log_diffs += (log_mag - prev_log_mag)
        prev_log_mag = log_mag
        count += 1

    return sum_log_diffs / count if count > 0 else 0.0

def sample_variance_for_radius(args):
    x, y, radius, samples, max_iter = args
    speeds = []
    for _ in range(samples):
        angle = np.random.rand() * 2 * np.pi
        radius_sample = radius * np.random.rand()
        dx = radius_sample * np.cos(angle)
        dy = radius_sample * np.sin(angle)
        
        # The key change is here: call the burning_ship_escape_speed function
        speed = burning_ship_escape_speed(x + dx, y + dy, max_iter)
        speeds.append(speed)
        
    speeds = np.clip(speeds, -1e10, 1e10)
    return float(np.var(speeds))

def boost_dynamic_range(image, alpha=0.2, beta=0.01):
    """
    Applies a custom non-linear normalization to boost dynamic range.
    Args:
        image: A numpy array of image data.
        alpha: Controls overall brightness boost.
        beta: Controls the contrast in mid-tones.
    Returns:
        A new numpy array with boosted dynamic range.
    """
    # Normalize to [0, 1] range first
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
    # Apply a non-linear transformation
    boosted_image = (image_norm ** alpha) * (image_norm + beta) / (image_norm + 1e-10)
    
    # Re-normalize to [0, 1] after the boost
    boosted_image = (boosted_image - boosted_image.min()) / (boosted_image.max() - boosted_image.min() + 1e-10)
    
    return boosted_image

if __name__ == "__main__":
    image = np.zeros((height, width, 3), dtype=np.float64)
    
    # Create a list of all tasks (one for each pixel)
    tasks = []
    for j in range(height):
        for i in range(width):
            x = center_x + (i - width // 2) * (scale / width)
            y = center_y + (j - height // 2) * (scale / height)
            tasks.append((x, y))

    # Define a new function that will be executed in parallel
    def process_pixel(pixel_coords):
        x, y = pixel_coords
        variances = []
        for r in radii:
            # You can keep the inner sampling loop
            speeds = []
            for _ in range(samples_per_radius):
                angle = np.random.rand() * 2 * np.pi
                radius_sample = r * np.random.rand()
                dx = radius_sample * np.cos(angle)
                dy = radius_sample * np.sin(angle)
                speed = mandelbrot_escape_speed(x + dx, y + dy, max_iter)
                speeds.append(speed)
            speeds = np.clip(speeds, -1e10, 1e10)
            variances.append(float(np.var(speeds)))
        return variances

    # Use the pool to process all tasks in parallel
    with Pool(processes=4) as pool:
        # Use imap for a more memory-efficient and progressive loop
        # Using a chunksize can also help
        results_iterator = pool.imap(process_pixel, tasks, chunksize=100)
        
        # Collect results and populate the image
        for k, variances in enumerate(tqdm(results_iterator, total=len(tasks))):
            j = k // width
            i = k % width
            image[j, i, :] = variances

    # Normalize and log transform
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply the new dynamic range boosting function
    boosted_image = boost_dynamic_range(image, alpha=0.2, beta=0.01)

    plt.imsave("burning_ship_variance_wall_art.jpg", boosted_image) # png gave too large a file for github

    #plt.imshow(boosted_image)
    #plt.axis('off')
    #plt.show()
