import cv2
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import measure
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
import math


# Function to compute all metrics
def compute_metrics(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    psnr_value = psnr(original, denoised)
    similarity, _ = ssim(original, denoised, full=True)
    uqi_value = uqi(original, denoised)
    fsim_value = fsim(original, denoised)
    dssim_value = 1 - similarity  # DSSIM is 1 - SSIM
    entropy_value = entropy(denoised)
    epr_value = edge_preservation_ratio(original, denoised)

    return mse, psnr_value, similarity, uqi_value, fsim_value, dssim_value, entropy_value, epr_value


# UQI - Universal Quality Index
def uqi(original, denoised):
    return np.mean((2 * original * denoised + 0.0001) / (original**2 + denoised**2 + 0.0001))


# FSIM - Feature Similarity Index
def fsim(original, denoised):
    # Placeholder implementation - replace with an actual FSIM calculation.
    return np.corrcoef(original.flatten(), denoised.flatten())[0, 1]


# Entropy
def entropy(image):
    return measure.shannon_entropy(image)


# Edge Preservation Ratio
def edge_preservation_ratio(original, denoised):
    edges_original = cv2.Canny(original, 100, 200)
    edges_denoised = cv2.Canny(denoised, 100, 200)
    edge_intersection = np.sum(np.logical_and(edges_original, edges_denoised))
    edge_union = np.sum(np.logical_or(edges_original, edges_denoised))

    return edge_intersection / edge_union if edge_union != 0 else 0


# Function to check edge preservation
def check_edge_preservation(original, denoised):
    edges_original = cv2.Canny(original, 100, 200)
    edges_denoised = cv2.Canny(denoised, 100, 200)

    edge_difference = cv2.absdiff(edges_original, edges_denoised)

    return edges_original, edges_denoised, edge_difference


# Function to plot noisy and denoised images
def plot_images(noisy, denoised):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    plt.title("Noisy Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    plt.title("Denoised Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Function to plot edge comparison
def plot_edges(edges_original, edges_denoised, edge_difference):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(edges_original, cmap="gray")
    plt.title("Edges: Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(edges_denoised, cmap="gray")
    plt.title("Edges: Denoised")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(edge_difference, cmap="gray")
    plt.title("Edge Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Main execution
image_path = "assets\\SpeckleSlobozia.png"
wavelet = 'db5'
threshold_scaling = 0.3
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Unable to load the image. Please check the file path.")
    exit()

image = image / 255.0

# Apply wavelet denoising
coeffs = pywt.wavedec2(image, wavelet, level=None)
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
threshold = threshold_scaling * sigma
thresholded_coeffs = [coeffs[0]]

for detail_coeffs in coeffs[1:]:
    thresholded_coeffs.append(tuple(
        pywt.threshold(c, value=threshold, mode='soft') for c in detail_coeffs
    ))

denoised_image = pywt.waverec2(thresholded_coeffs, wavelet)
denoised_image = np.clip(denoised_image, 0, 1) * 255
denoised_image = denoised_image.astype(np.uint8)

image = (image * 255).astype(np.uint8)

print(f"Original: dtype={image.dtype}, shape={image.shape}")
print(f"Denoised: dtype={denoised_image.dtype}, shape={denoised_image.shape}")

# Compute and print metrics
mse, psnr_value, similarity, uqi_value, fsim_value, dssim_value, entropy_value, epr_value = compute_metrics(image, denoised_image)
print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {similarity:.4f}")
print(f"UQI: {uqi_value:.4f}")
print(f"FSIM: {fsim_value:.4f}")
print(f"DSSIM: {dssim_value:.4f}")
print(f"Entropy: {entropy_value:.4f}")
print(f"EPR: {epr_value:.4f}")

# Check edge preservation
edges_image, edges_denoised_image, edge_difference = check_edge_preservation(image, denoised_image)

# Plot images and edges
plot_images(image, denoised_image)
plot_edges(edges_image, edges_denoised_image, edge_difference)
