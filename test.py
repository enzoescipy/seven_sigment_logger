import numpy as np
import cv2
src_pts = [[8, 136], [415, 52], [420, 152], [14, 244]]
def four_point_transform(img, src_pts):

    def get_euler_distance(pt1, pt2):
        return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5

    src_pts = np.array(src_pts, dtype=np.float32)
    width = get_euler_distance(src_pts[0], src_pts[1])
    height = get_euler_distance(src_pts[0], src_pts[3])
    print(width, height)

    dst_pts = np.array([[0, 0],   [width, 0],  [width, height], [0, height]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(img, M, (width, height))
    return warp

four_point_transform(None, src_pts)