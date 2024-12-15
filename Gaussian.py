import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def compute_metrics(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    psnr = cv2.PSNR(original, denoised)
    similarity, _ = ssim(original, denoised, full=True, channel_axis=2)
    return mse, psnr, similarity

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


image_path = "assets\\photono1.png"
# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load the image. Please check the file path.")
    exit()

# kernel size = (5, 5), sigma = 0
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

mse, psnr, similarity = compute_metrics(image, smoothed_image)
print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {similarity:.4f}")
edges_image, edges_denoised_image, edge_difference = check_edge_preservation(image, smoothed_image)

plot_images(image, smoothed_image)
plot_edges(edges_image, edges_denoised_image, edge_difference)