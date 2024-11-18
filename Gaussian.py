import cv2
import numpy as np

def denoise_with_gaussian(image_path, kernel_size=(5, 5), sigma=0):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image. Please check the file path.")
        return


    smoothed_image = cv2.GaussianBlur(image, kernel_size, sigma)

    cv2.imshow("Original Image", image)
    cv2.imshow("Gaussian Smoothed Image", smoothed_image)

    print("Press any key to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    image_path = "assets\\photono1.png"
    denoise_with_gaussian(image_path, kernel_size=(5, 5), sigma=0)