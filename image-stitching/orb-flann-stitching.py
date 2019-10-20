import cv2
import numpy as np


## RESOURCES
# tutorial: https://medium.com/pylessons/image-stitching-with-opencv-and-python-1ebd9e0a6d78
# opencv feature matching docs: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# cv2.findHomography(): https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findhomography
# cv2.warpPerspective(): https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20warpPerspective(InputArray%20src,%20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int%20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue)
# query vs. train: https://subscription.packtpub.com/book/application_development/9781785283932/6/ch06lvl1sec57/creating-the-panoramic-image
# cropping: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy

## FUNCTIONS
def draw_keypoints(img, keypoints):
    for kp in keypoints:
            x, y = kp.pt         
            cv2.circle(img, (int(x), int(y)), 2, (0,255,0) )


## READ IN IMAGES
img_query = cv2.imread("images/skier/skier-left.jpg",1) # query, = the first/original image, what we will stitch TO
img_train = cv2.imread("images/skier/skier-right.jpg",1) # train = the second/additional image, we will distort/crop using the homography matrix

# display images, for testing purposes
cv2.imshow("Query Image", img_query)
cv2.imshow("Train Image", img_train)
cv2.waitKey(0)


## FIND KEYPOINTS AND DESCRIPTORS --> ORB
# initiate detector
orb = cv2.ORB_create(nfeatures=5000) # increasing nfeatures allows for more keypoints but takes longer to compute

# calculate keypoints and descriptors
kp_query, des_query = orb.detectAndCompute(img_query, None)
kp_train, des_train = orb.detectAndCompute(img_train, None)

# draw keypoints, for testing purposes
draw_query_kp = img_query.copy()
draw_train_kp = img_train.copy()
draw_keypoints(draw_query_kp, kp_query)
draw_keypoints(draw_train_kp, kp_train)
cv2.imshow("Query Keypoints", draw_query_kp)
cv2.imshow("Train Keypoints", draw_train_kp)
cv2.waitKey(0)


## FIND MATCHES --> FLANN + knn with ratio test
# set up matching parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12, 6
                   key_size = 12,     # 20, 12
                   multi_probe_level = 1) #2, 1
search_params = dict(checks=50)   # number of times to traverse tree, larger is more accurate but takes longer

# find matches
flann = cv2.FlannBasedMatcher(index_params,search_params)
flann_rawMatches = flann.knnMatch(des_query,des_train,k=2)

# apply ratio test to filter out poor matches
flann_matches = []
flann_ratio = 0.25 # increase to allow more matches, but may lose accuracy

for m,n in flann_rawMatches:
    if m.distance < flann_ratio*n.distance:
        flann_matches.append(m)

# draw matches, for testing purposes
draw_params = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)
drawn_matches = cv2.drawMatches(img_query, kp_query, img_train, kp_train, flann_matches, None, **draw_params)

# display matches, for testing purposes
cv2.imshow("Feature Matches", drawn_matches)
cv2.waitKey(0)

## CALCULATE HOMOGRAPHY MATRIX
MIN_MATCH_COUNT = 4 # computing a homography matrix requires at least 4 matches

if len(flann_matches) > MIN_MATCH_COUNT: # make sure there are enough matches 

    # points of interest
    queryPts = np.float32([ kp_query[m.queryIdx].pt for m in flann_matches ]).reshape(-1,1,2)
    trainPts = np.float32([ kp_train[m.trainIdx].pt for m in flann_matches ]).reshape(-1,1,2)

    # calculate homography
    reprojThresh = 5 # reprojection error for RANSAC to treat a point as an inlier
    M, status = cv2.findHomography(trainPts, queryPts, cv2.RANSAC, reprojThresh) # use RANSAC for robust linear regression

    ## WARP TRAIN IMAGE USING HOMOGRAPHY
    # output dimensions for warping
    width = img_query.shape[1] + img_train.shape[1] # width is at most both images completely side by side, this will be cropped for final output
    height = img_query.shape[0] + img_train.shape[0] # height is at most both images stacked on top of each other, this will be cropped for final output 

    # apply homography warping
    result = cv2.warpPerspective(img_train, M, (width, height)) # train image plan will be warped by the homography to the query image plane
    
    # display warped image, for testing purposes
    cv2.imshow("Warped Train Image", result)
    cv2.waitKey(0)

    # fill in result with query image
    result[0:img_query.shape[0], 0:img_query.shape[1]] = img_query 
    
    # display raw stitching, for testing purposes
    cv2.imshow("Raw Stitching", result)
    cv2.waitKey(0)

    ## CROP LEFT-OVER BLACK SPACE
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    mask = gray > 0 # mask of non-black pixels (assuming image has a single channel)

    coords = np.argwhere(mask) # coordinates of non-black pixels

    x0, y0 = coords.min(axis=0) # bounding box of non-black pixels
    x1, y1 = coords.max(axis=0) + 1 # slices are exclusive at the top

    output = result[x0:x1, y0:y1] # get the contents of the bounding box

    # display final cropped stitching, for testing purposes
    cv2.imshow("Final Output", output) 
    cv2.waitKey(0)

else: # not enough matches to compute homography matrix
    print("Need at least", MIN_MATCH_COUNT, "matches, found:", (len(flann_matches)/MIN_MATCH_COUNT))










