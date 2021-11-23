import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("picture396.jpg")
img2 = cv2.imread("picture397.jpg")

def getMapimgFullHD(img):
    return img[610:1030,60:470]
img1 = getMapimgFullHD(img1)
img2 = getMapimgFullHD(img2)

akaze = cv2.AKAZE_create()

kp1, des1 = akaze.detectAndCompute(img1,None)
kp2, des2 = akaze.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

ref_matched_kpts = np.float32(
    [kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
sensed_matched_kpts = np.float32(
    [kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, status = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

warped_image = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

cv2.imwrite("warped3.jpg",warped_image)
class ImageRegistration:
    def __init__(self,map_scope_xy:list) -> None:
        self.detecter = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher()
        self.x1, self.x2 = map_scope_xy[0], map_scope_xy[1]
        self.y1, self.y2 = map_scope_xy[2], map_scope_xy[3]
    
    def execute(self, img1:np.ndarray, img2:np.ndarray)->np.ndarray:
        img1 = img1[self.x1:self.y1,self.x2:self.y2]
        img2 = img2[self.x1:self.y1,self.x2:self.y2]
        kp1, des1 = self.detecter.detectAndCompute(img1,None)
        kp2, des2 = self.detecter.detectAndCompute(img2,None)
        matches = self.bf.knnMatch(des1,des2,k=2)
        good_matches = [
            [m] for m, n in matches if m.distance < 0.75 * n.distance
        ]
        ref_matched_kpts = np.float32(
            [kp1[m[0].queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        sensed_matched_kpts = np.float32(
            [kp2[m[0].trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

        warped_image = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
        return warped_image
