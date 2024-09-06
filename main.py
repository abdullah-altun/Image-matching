from subpixelEedges.main import edgeDetection
from processing import processing
import ICT
from python_src import test
import cv2

path = "images/camera/Image_20240817115736082.jpg"
edgeDetection(path)
img = processing(f"images/Edge/{path.split('/')[-1]}")
cv2.imwrite(f"images/camera_input/{path.split('/')[-1]}",img)

test.main(f"images/camera_input/{path.split('/')[-1]}",f"images/camera_input/{path.split('/')[-1]}")
img = processing(f"images/Subpixel/{path.split('/')[-1]}")
cv2.imwrite(f"images/camera_input2/{path.split('/')[-1]}",img)

ICT.main(f"images/camera_input2/{path.split('/')[-1]}","camera_calib/17_5.json")
