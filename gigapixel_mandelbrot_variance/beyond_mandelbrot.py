
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from multiprocessing import Pool
from tqdm import tqdm

# Parameters
width, height = 8192, 8192
center_x, center_y = -1.875, 1.125
tile_num = 1

# tiles 8192x8192
"""
start -0.75, 0.0
scale 3
tile size 0.75
half is 0.375

top left corner -2.25, 1.5

scale 0.75 for gigapixel tiling
can run up to 5 in parallel on kaggle
should take about 2 days in parallel on kaggle

tile 1 center: -1.875, 1.125
tile 2 center: -1.125, 1.125
tile 3 center: -0.375, 1.125
tile 4 center: -0.375, 1.125

tile 5 center: -1.875, 0.375
tile 6 center: -1.125, 0.375
tile 7 center: -0.375, 0.375
tile 8 center: -0.375, 0.375

tile 9 center: -1.875, -0.375
tile 10 center: -1.125, -0.375
tile 11 center: -0.375, -0.375
tile 12 center: -0.375, -0.375

tile 13 center: -1.875, -1.125
tile 14 center: -1.125, -1.125
tile 15 center: -0.375, -1.125
tile 16 center: -0.375, -1.125
"""

scale = 3.0  # Reduced scale to zoom in and capture more detail
max_iter = 10
samples_per_radius = 10
radii = [1e-4, 1e-5, 1e-6]

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
        z = z*z + c
        
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
    pool = Pool(processes=4)

    for j in tqdm(range(height)):
        for i in range(width):
            x = center_x + (i - width // 2) * (scale / width)
            y = center_y + (j - height // 2) * (scale / height)

            tasks = [(x, y, r, samples_per_radius, max_iter) for r in radii]
            variances = pool.map(sample_variance_for_radius, tasks)
            image[j, i, :] = variances

    pool.close()
    pool.join()

    # Normalize and log transform
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply the new dynamic range boosting function
    boosted_image = boost_dynamic_range(image, alpha=0.2, beta=0.01)

    plt.imsave("mandelbrot_tile_" + tile_num + ".jpg", boosted_image)
    plt.imshow(boosted_image)
    plt.axis('off')
    plt.show()
