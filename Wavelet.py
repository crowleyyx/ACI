import cv2
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt

def wavelet_denoising(image_path, wavelet='db1', threshold_scaling=0.1):


    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load the image. Please check the file path.")
        return


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

    image_path = "assets\\photono1.png"  
    wavelet_denoising(image_path, wavelet='db1', threshold_scaling=0.1)