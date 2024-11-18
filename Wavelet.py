import cv2
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt

def wavelet_denoising(image_path, wavelet='db1', threshold_scaling=0.1):

    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load the image. Please check the file path.")
        return

    # Normalize the image
    image = image / 255.0  # Scale pixel values to [0, 1]

    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=None)  # Decompose to wavelet coefficients

    # Determine threshold
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimate noise standard deviation
    threshold = threshold_scaling * sigma

    # Thresholding: Apply soft thresholding to detail coefficients
    thresholded_coeffs = [coeffs[0]]  # Keep the approximation coefficients
    for detail_coeffs in coeffs[1:]:
        thresholded_coeffs.append(tuple(
            pywt.threshold(c, value=threshold, mode='soft') for c in detail_coeffs
        ))

    # Reconstruct the image from thresholded coefficients
    denoised_image = pywt.waverec2(thresholded_coeffs, wavelet)

    # Clip values to valid range and rescale to [0, 255]
    denoised_image = np.clip(denoised_image, 0, 1) * 255
    denoised_image = denoised_image.astype(np.uint8)

    # Display the original and denoised images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image (Wavelet Thresholding)")
    plt.imshow(denoised_image, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify the image path
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    wavelet_denoising(image_path, wavelet='db1', threshold_scaling=0.1)