from compare_contoures import ContourComparer
from dfx_to_contour import Dxf2ContourReader
import cv2
import json
import numpy as np
from PIL import Image, ImageTk

thresh = 30
thresh_mode = 1
img_ppmm = None
max_dist_in_mm = 1
max_dist_erroneous = False
img_preview_shape = None
mdl_ppmm = 11.656380890724625

try:
    max_dist_in_mm = float(max_dist_in_mm)
except ValueError:
    max_dist_erroneous = True

comparer = ContourComparer()
reader = Dxf2ContourReader(inputMask=True)

def load_model():
       
    filename = "test_files/002.dxf"
    
    model_loaded = True

    reader.read_file(filename, mdl_ppmm)
    model_contours, _ = reader.get_contours()
    comparer.set_model_contours(model_contours)



def render_image(image=None, max_dim=None):
        h, w = image.shape
        if h > max_dim or w > max_dim:
            fx = max_dim / w
            fy = max_dim / h
            scale = min(fx, fy)
            preview_img = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            preview_img = image
        pil_img = Image.fromarray(preview_img)
        render = ImageTk.PhotoImage(pil_img)
        return render

def find_and_filter_contours(bimg):
    n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(bimg)
    if n_blobs > 2:
        max_area = np.max(stats[1:, -1])
        if np.sum(stats[1:, -1] > 0.2 * max_area) > 2:
            raise RuntimeError("There seems to be multiple objects in the image. " +
                                "This program is for checking a single part only.")
        new_bimg = np.zeros_like(bimg)
        max_idx = np.where(stats[1:, -1] == max_area)[0][0] + 1
        new_bimg[labels == max_idx] = 255
        bimg = new_bimg

    cnts, hier = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return cnts, hier


def calib_json_read(filename):
    with open(filename) as json_file:
        json_data = json.load(json_file)
        json_file.close()

    cal_data = {}
    file_ok = True
    try:
        cal_data["camera_matrix"] = np.asarray(json_data["camera_matrix"])
        cal_data["distortion_coeffs"] = np.asarray(json_data["distortion_coeffs"])
        cal_data["new_camera_matrix"] = np.asarray(json_data["new_camera_matrix"])
        cal_data["perspective_transformation"] = np.asarray(json_data["perspective_transformation"])
        ppmm = json_data["ppmm"]
    except:
        file_ok = False
    else:
        if (cal_data["camera_matrix"].shape != (3, 3) or
            cal_data["distortion_coeffs"].shape != (1, 5) or
            cal_data["new_camera_matrix"].shape != (3, 3) or
            cal_data["perspective_transformation"].shape != (3, 3) or
            not(isinstance(ppmm, (int, float)))):
            file_ok = False
    
    if file_ok:
        img_ppmm = ppmm
        return cal_data,img_ppmm
    
def main(path,calib_path):
    load_model()
    img = cv2.imread(path,0)
    img[img<10] = 0; img[img>9] = 255
    calib_data,img_ppmm = calib_json_read(calib_path)

    _, test_bw = cv2.threshold(img, thresh, 255, thresh_mode)

    contours, _ = cv2.findContours(test_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) == 0 or 
                    cv2.contourArea(max(contours, key=cv2.contourArea)) < 400):
        print("Error!!!!!")
    else:
        img_ud = img
        _, bw = cv2.threshold(img_ud, thresh, 255, thresh_mode)

        try:
            contours, _ = find_and_filter_contours(bw)
        except RuntimeError as re:
            raise ValueError("Hataaa!!!!!!")
        else:
            output_img = np.zeros_like(bw) 

            # Konturları görüntüye çiz
            cv2.drawContours(output_img, contours, -1, (255, 255, 255), 1)  #Konturları beyaz renkte çiz

            cv2.imshow("Contours", output_img)
            cv2.imwrite("contours_output.jpg", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            comparer.match_contour_to_model(contours, max_dist_in_mm * img_ppmm, 
                                                img_ppmm, mdl_ppmm, 
                                                img_ud)