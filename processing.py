import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def find_endpoints(bw):
    res_bw = np.zeros_like(bw)
    strels = [np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]]), 
                np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]]),
                np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]]),
                np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]]),
                np.array([[0, -1, -1], [1, 1, -1], [0, -1, -1]]),
                np.array([[0, 1, 0], [-1, 1, -1], [-1, -1, -1]]),
                np.array([[-1, -1, 0], [-1, 1, 1], [-1, -1, 0]]),
                np.array([[-1, -1, -1], [-1, 1, -1], [0, 1, 0]])]
    for strel in strels:
        tmp = cv2.morphologyEx(bw, cv2.MORPH_HITMISS, strel)
        res_bw += tmp
    
    end_ys, end_xs = np.where(res_bw == 255)
    end_points = np.vstack((end_xs, end_ys)).T

    return end_points

def check_gaps(bw, spat_reso):
    end_ps = find_endpoints(bw)
    pairs = []
    if end_ps.shape[0] > 1:
        while end_ps.shape[0] > 1:
            prev_len = end_ps.shape[0]
            dists = end_ps[1:] - end_ps[0]
            dists = np.sqrt(dists[:, 0]**2 + dists[:, 1]**2)
            if min(dists) < 2 * spat_reso:
                i = np.argmin(dists) + 1
                pairs.append(([end_ps[0], end_ps[i]]))
                end_ps = np.delete(end_ps, [0, i], axis=0)
            else:
                end_ps = end_ps[1:]
    return pairs

def processing(path):
    img = cv2.imread(path,0)
    name = path.split("/")[-1]
    img_array = img.copy()
    img_array[img<11] = 0
    img_array[img>11] = 255
    img = img_array

    _, img_thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = contour[:, 0, :]  # Konturu (x, y) noktalarına ayırma
        tck, u = splprep([contour[:, 0], contour[:, 1]], s=1.0)
        x_new, y_new = splev(u, tck)
        smoothed_contour = np.array([x_new, y_new]).T.astype(int)
        cv2.polylines(img_thresh, [smoothed_contour], isClosed=True, color=255, thickness=1)

    img_array = img_thresh.copy()
    img_array[img_thresh == 255] = 0
    img_array[img_thresh == 0] = 255
    img_thresh = img_array

    pairs = check_gaps(img_thresh, 5)
    if pairs != []:
        for pair in pairs:
            cv2.line(img_thresh, tuple(pair[0]), tuple(pair[1]), 255)

    h, w = img_thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    seed_point = (w // 2, h // 2)
    cv2.floodFill(img_thresh, mask, seed_point, 255)
    return img_thresh
    