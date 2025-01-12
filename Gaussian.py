import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from piq import FSIMLoss
import torch
import matplotlib.pyplot as plt

def compute_metrics(original, denoised):
    # Convert images to PyTorch tensors for FSIM
    original_tensor = torch.from_numpy(original).permute(2, 0, 1).float() / 255.0
    denoised_tensor = torch.from_numpy(denoised).permute(2, 0, 1).float() / 255.0

    # Mean Squared Error
    mse = np.mean((original - denoised) ** 2)
    # Peak Signal-to-Noise Ratio
    psnr = cv2.PSNR(original, denoised)
    # Structural Similarity Index
    similarity, _ = ssim(original, denoised, full=True, channel_axis=2)
    # Feature Similarity Index Measure (FSIM)
    fsim_loss = FSIMLoss(data_range=1.0)
    fsim_score = 1 - fsim_loss(original_tensor.unsqueeze(0), denoised_tensor.unsqueeze(0)).item()
    # Shannon Entropy Difference
    entropy_original = shannon_entropy(original)
    entropy_denoised = shannon_entropy(denoised)
    entropy_diff = abs(entropy_original - entropy_denoised)
    
    return mse, psnr, similarity, fsim_score, entropy_diff

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
image_path = "assets\\SaltPepperSlobozia.png"
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load the image. Please check the file path.")
    exit()

# Apply Gaussian Blur for denoising
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

# Compute metrics
mse, psnr, similarity, fsim_score, entropy_diff = compute_metrics(image, smoothed_image)
print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {similarity:.4f}")
print(f"FSIM: {fsim_score:.4f}")
print(f"Entropy Difference: {entropy_diff:.4f}")

# Check edge preservation
edges_image, edges_denoised_image, edge_difference = check_edge_preservation(image, smoothed_image)

# Plot the denoised image and edge difference
plot_images(image, smoothed_image)
plot_edges(edges_image, edges_denoised_image, edge_difference)
