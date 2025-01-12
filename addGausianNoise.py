import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to choose a file
def choose_file():
    Tk().withdraw()  # Close the root window
    file_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return file_path

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, stddev=25):
    gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to add Salt-and-Pepper noise
def add_salt_and_pepper(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    total_pixels = image.size // image.shape[-1]

    # Add salt noise
    num_salt = int(total_pixels * salt_prob*5)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[salt_coords[0], salt_coords[1]] = [255, 255, 255]

    # Add pepper noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[pepper_coords[0], pepper_coords[1]] = [0, 0, 0]

    return noisy

# Function to add Speckle noise
def add_speckle_noise(image):
    speckle_noise = np.random.normal(0, 1, image.shape).astype(np.float32)
    noisy_image = image + image * speckle_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Main program
file_path = choose_file()
if file_path:
    # Load the selected image
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prompt the user to choose a noise type
    print("Choose the type of noise to add:")
    print("1: Gaussian Noise")
    print("2: Salt-and-Pepper Noise")
    print("3: Speckle Noise")
    choice = input("Enter the number (1/2/3): ")

    if choice == "1":
        noisy_image = add_gaussian_noise(image)
        noise_type = "Gaussian Noise"
    elif choice == "2":
        noisy_image = add_salt_and_pepper(image)
        noise_type = "Salt-and-Pepper Noise"
    elif choice == "3":
        noisy_image = add_speckle_noise(image)
        noise_type = "Speckle Noise"
    else:
        print("Invalid choice. Exiting.")
        exit()

    # Show the original and noisy images
    #cv2.imshow("Original Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow(noise_type, cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No file selected.")
