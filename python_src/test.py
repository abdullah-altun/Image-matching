
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
import cv2
from python_src.devernay_edges import DevernayEdges


def main(path1,path2):
    sigma = 1.0
    high_treshold = 5.0
    low_threshold = 1.0

    image_rgb = Image.open(path1)    
    # image_rgb = Image.open("dog.jpg")

    # convert to binary
    image_binary = Image.open(path2)
    # image_binary = Image.open("binary_dog_image.jpg")

    devernayEdges = DevernayEdges(image_binary, sigma, high_treshold, low_threshold)
    [edges_x, edges_y] = devernayEdges.detect_edges()
    
    print(f"edges_x len: {len(edges_x)}")
    print(f"edges_y len: {len(edges_y)}")
    """
    plt.figure(1)
    plt.title("Devernay Edge Detection")
    plt.imshow(image_rgb)
    plt.scatter(edges_x, edges_y, color="magenta", marker=".", linewidth=.1)
    plt.show()
    """

    blank_image = np.zeros_like(image_rgb)
    for i in range(len(edges_x)):
        cv2.circle(blank_image, (int(edges_x[i]), int(edges_y[i])), 1, (255, 255, 255), -1)
    cv2.imwrite(f"images/Subpixel/{path1.split('/')[-1]}",blank_image)