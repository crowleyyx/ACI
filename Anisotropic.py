import cv2
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import matplotlib.pyplot as plt

def anisotropic_denoising(image_path, weight=0.1, multichannel=True):

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image. Please check the file path.")
        return


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

 
    denoised_image = denoise_tv_chambolle(image_rgb, weight=weight)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Anisotropic Denoising")
    plt.imshow(denoised_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    image_path = "assets\\photono1.png"  
    anisotropic_denoising(image_path)