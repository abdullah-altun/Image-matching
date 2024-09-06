import cv2 
import numpy as np
img = cv2.imread("images/Edge/den2.jpg")

print(np.unique(img))

img_copy = img.copy()
img_copy[img<11] = 255
img_copy[img>10] = 0
cv2.imwrite("images1.jpg",img_copy)