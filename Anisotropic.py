import cv2
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import img_as_float
from skimage.measure import shannon_entropy


def compute_metrics(original, denoised):
    # Convert images to float type for better precision in calculations
    original_float = img_as_float(original)
    denoised_float = img_as_float(denoised)

    mse = np.mean((original - denoised) ** 2)
    psnr = cv2.PSNR(original, denoised)
    similarity, _ = ssim(original, denoised, full=True, channel_axis=2)
    uqi = compute_uqi(original, denoised)
    fsim_value = compute_fsim(original, denoised)
    dssim_value = 1 - similarity  # DSSIM is 1 - SSIM

    # Compute entropy
    entropy_original = shannon_entropy(original_float)
    entropy_denoised = shannon_entropy(denoised_float)

    return mse, psnr, similarity, uqi, fsim_value, dssim_value, entropy_original, entropy_denoised

def compute_uqi(original, denoised):
    # This is a simplified implementation of the Universal Image Quality Index (UQI)
    original_float = img_as_float(original)
    denoised_float = img_as_float(denoised)
    numerator = 4 * np.mean(original_float * denoised_float) * np.mean(original_float) * np.mean(denoised_float)
    denominator = (np.mean(original_float) + np.mean(denoised_float)) * (np.std(original_float) + np.std(denoised_float))
    return numerator / denominator

def compute_fsim(original, denoised):
    # Feature Similarity Index (FSIM) - we will approximate this by considering edges and gradients
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    original_edges = canny(original_gray)
    denoised_edges = canny(denoised_gray)
    
    # FSIM is based on edge similarity, so we compute the overlap of edge pixels
    edge_similarity = np.sum(original_edges & denoised_edges) / np.sum(original_edges | denoised_edges)
    return edge_similarity

def check_edge_preservation(original, denoised):
    edges_original = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 100, 200)
    edges_denoised = cv2.Canny(cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY), 100, 200)

    edge_difference = cv2.absdiff(edges_original, edges_denoised)

    return edges_original, edges_denoised, edge_difference

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

# Load the image
image_path = "assets\\photono1.png"
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load the image. Please check the file path.")
    exit()

# Convert image to RGB (for the denoising process)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform denoising using TV Chambolle
denoised_image = denoise_tv_chambolle(image_rgb, weight=0.1)
denoised_image = (denoised_image * 255).astype(np.uint8)

# Compute metrics
mse, psnr, similarity, uqi, fsim_value, dssim_value, entropy_original, entropy_denoised = compute_metrics(image, denoised_image)

# Print the results
print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {similarity:.4f}")
print(f"UQI: {uqi:.4f}")
print(f"FSIM: {fsim_value:.4f}")
print(f"DSSIM: {dssim_value:.4f}")
print(f"Entropy (Original): {entropy_original:.4f}")
print(f"Entropy (Denoised): {entropy_denoised:.4f}")

# Edge preservation analysis
edges_image, edges_denoised_image, edge_difference = check_edge_preservation(image, denoised_image)

# Plot images and edges
plot_images(image, denoised_image)
plot_edges(edges_image, edges_denoised_image, edge_difference)
