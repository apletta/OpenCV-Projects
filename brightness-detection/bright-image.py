import numpy as np 
import argparse
import cv2

### resources
#https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/

# argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-r", "--radius", type=int, help="radius of Gaussian blur; must be odd")
ap.add_argument("-t", "--threshold", type=int, help="threshold value for detecting brightness")

args = vars(ap.parse_args())

maxThresh = args["threshold"] # threshold for bright spot detection


# load image and convert to grayscale
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#cv2.imshow("hsv", hsv)
#cv2.waitKey(0)

############### this is naive because there is no pre-processing, so very susceptible to any noise

# find brightest are of image (largest intensity value)
#(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray) 
#cv2.circle(image, maxLoc, args["radius"], (255, 0, 0), 2)

# display results
#cv2.imshow("Naive", image) 
#cv2.waitKey(0)

###############

# apply Gaussian blur to image before finding bright region
gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
#cv2.imshow("second gray", gray)
#cv2.waitKey(0)

(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
image = orig.copy()
if maxVal >= maxThresh:
	cv2.circle(image, maxLoc, args["radius"], (255, 0, 0), 2)

print("minVal", minVal, "|", "maxVal", maxVal)

# display new results
cv2.imshow("Robust", image)
cv2.waitKey(0)
