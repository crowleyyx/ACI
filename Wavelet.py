import cv2
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def compute_metrics(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    psnr = cv2.PSNR(original, denoised)
    similarity, _ = ssim(original, denoised, full=True)
    return mse, psnr, similarity

def check_edge_preservation(original, denoised):
    edges_original = cv2.Canny(original, 100, 200)
    edges_denoised = cv2.Canny(denoised, 100, 200)

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
wavelet='db1'
threshold_scaling=0.1
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable to load the image. Please check the file path.")
    exit()


image = image / 255.0


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


mse, psnr, similarity = compute_metrics(image, denoised_image)
print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {similarity:.4f}")
edges_image, edges_denoised_image, edge_difference = check_edge_preservation(image, denoised_image)

plot_images(image, denoised_image)
plot_edges(edges_image, edges_denoised_image, edge_difference)