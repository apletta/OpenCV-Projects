import numpy as np
import cv2
from matplotlib import pyplot as plt

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html

img_left = cv2.imread("images/skier/skier-left.jpg",0)
img_right = cv2.imread("images/skier/skier-right.jpg",0)

# Initiate detector
orb = cv2.ORB_create(nfeatures=1000)

# # find the keypoints with ORB
# kp = orb.detect(img,None)

# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)


keypoints_left, descriptors_left = orb.detectAndCompute(img_left, None)
keypoints_right, descriptors_right = orb.detectAndCompute(img_right, None)


# draw only keypoints location,not size and orientation
keypointColor=(0, 255, 0)
img2_left = cv2.drawKeypoints(img_left,keypoints_left, outImage=np.array([]), color=keypointColor, flags=0)
img2_right = cv2.drawKeypoints(img_right,keypoints_right, outImage=np.array([]), color=keypointColor, flags=0)

fig, axarr = plt.subplots(1,2)
fig.suptitle("ORB Image Keypoints")

ax1 = axarr[0]
ax1.set_title("Left Image")
ax1.imshow(img2_left)


ax2 = axarr[1]
ax2.set_title("Right Image")
ax2.imshow(img2_right)

plt.show()



# find the keypoints and descriptors with SIFT
