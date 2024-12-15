import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim


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


# Load an image
image = cv2.imread("assets\\photono1.png")
if image is None:
    print("Error loading image.")
    exit()

dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

mse, psnr, similarity = compute_metrics(image, dst)
print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {similarity:.4f}")
edges_image, edges_denoised_image, edge_difference = check_edge_preservation(image, dst)

plot_images(image, dst)
plot_edges(edges_image, edges_denoised_image, edge_difference)