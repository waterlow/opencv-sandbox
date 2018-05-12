import numpy as np
import cv2
import itertools

img = cv2.imread('image0002.jpg', 1)

height = int(img.shape[0]/2)
width = int(img.shape[1]/2)
img = cv2.resize(img, (width, height))

lines = img.copy()
canny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.GaussianBlur(canny, (11,11), 0)
canny = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY_INV)[1]
canny = cv2.Canny(canny, 50, 100)

cnts = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
cnts.sort(key=cv2.contourArea, reverse=True)

warp = []
for i, c in enumerate(cnts):
    arclen = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.1*arclen, True)

    if len(approx) == 4 and warp == []:
        warp = approx.copy()

points = warp[:,0,:]
points = sorted(points, key=lambda x:x[1])  # yが小さいもの順に並び替え。
top = sorted(points[:2], key=lambda x:x[0])  # 前半二つは四角形の上。xで並び替えると左右も分かる。
bottom = sorted(points[2:], key=lambda x:x[0], reverse=True)  # 後半二つは四角形の下。同じくxで並び替え。
points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。

width = max(np.sqrt(((points[0][0]-points[2][0])**2)*2), np.sqrt(((points[1][0]-points[3][0])**2)*2))
height = max(np.sqrt(((points[0][1]-points[2][1])**2)*2), np.sqrt(((points[1][1]-points[3][1])**2)*2))
dst = np.array([np.array([0, 0]), np.array([width-1, 0]), np.array([width-1, height-1]), np.array([0, height-1])], np.float32)

trans = cv2.getPerspectiveTransform(points, dst)
warped = cv2.warpPerspective(img, trans, (int(width), int(height)))

cv2.imshow('warp', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
