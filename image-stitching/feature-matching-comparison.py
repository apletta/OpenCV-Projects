import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html


def draw_keypoints(vis, keypoints, color):
    for kp in keypoints:
            x, y = kp.pt         
            cv2.circle(vis, (int(x), int(y)), 2, color)

img_left = cv2.imread("images/skier/skier-left.jpg",1)
img_right = cv2.imread("images/skier/skier-right.jpg",1)


## ORB
# Initiate detector
orb = cv2.ORB_create(nfeatures=5000)

keypoints_left, descriptors_left = orb.detectAndCompute(img_left, None)
keypoints_right, descriptors_right = orb.detectAndCompute(img_right, None)


# draw only keypoints location,not size and orientation
keypointColor=(0, 255, 0)
drawLeft = img_left.copy()
drawRight = img_right.copy()
draw_keypoints(drawLeft, keypoints_left, color=keypointColor)
draw_keypoints(drawRight, keypoints_right, color=keypointColor)


## CROSS CHECK MATCHING (essentially, use ratio=1)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(descriptors_left,descriptors_right)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first N matches, not very dynamic
N = 150
cc_matches_img = cv2.drawMatches(img_left,keypoints_left,img_right,keypoints_right,matches[:N],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0,255,0))

## BRUTE FORCE + KNN MATCHING WITH RATIO TEST (most accurate, slowest)
# get matches
bf = cv2.BFMatcher()
rawMatches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

# apply ratio test
matches = []
ratio = 0.25
for m,n in rawMatches:
	if m.distance < ratio*n.distance:
		matches.append(m)

matches_for_img = [matches]
knn_matches_img = cv2.drawMatchesKnn(img_left,keypoints_left,img_right,keypoints_right,matches_for_img,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0,255,0))

## FLANN + KNN MATCHING WITH RATIO TEST (not as accurate as bf, but fastest)
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12, 6
                   key_size = 12,     # 20, 12
                   multi_probe_level = 1) #2, 1
search_params = dict(checks=50)   # number of times to traverse tree, larger is more accurate but takes longer
flann = cv2.FlannBasedMatcher(index_params,search_params)
flann_rawMatches = flann.knnMatch(descriptors_left,descriptors_right,k=2)

# apply ratio test
flann_matches = []
flann_ratio = 0.25

for m,n in flann_rawMatches:
	if m.distance < ratio*n.distance:
		flann_matches.append(m)

flann_matches_for_img = [flann_matches]
flann_matches_img = cv2.drawMatchesKnn(img_left,keypoints_left,img_right,keypoints_right,flann_matches_for_img,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0,255,0))

f, axarr = plt.subplots(1,3)
f.suptitle("Feature Matching Techniques using ORB")
axarr[0].imshow(knn_matches_img)
axarr[0].set_title("Brute Force + Knn Matches, ratio = "+str(ratio))
axarr[1].imshow(flann_matches_img)
axarr[1].set_title("FLANN + Knn Matches, ratio = "+str(flann_ratio))
axarr[2].imshow(cc_matches_img)
axarr[2].set_title("Cross-Check Matches, first "+str(N)+" matches")

for i in range(0,3): # clear axis labels on plots for easier analysis
	axarr[i].get_yaxis().set_visible(False)
	axarr[i].get_xaxis().set_visible(False)

plt.show()




