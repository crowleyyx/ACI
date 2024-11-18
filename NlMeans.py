import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
# Load an image
image = cv.imread("assets\\photono1.png")
if image is None:
    print("Error loading image.")
    exit()



dst = cv.fastNlMeansDenoisingColored(image,None,10,10,7,21)

plt.subplot(121),plt.imshow(image)
plt.subplot(122),plt.imshow(dst)
plt.show()