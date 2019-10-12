import numpy as np 
import argparse
import cv2

### Resources
# tutorial -->  https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
# paper    -->  http://matthewalunbrown.com/papers/ijcv2007.pdf


### Functions
def scaleImage(image, width): # width in pixels
	ratio = width / image.shape[1] 
	dim = (width, int(image.shape[0]*ratio))

	resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA) # https://stackoverflow.com/questions/33183272/how-to-resize-image-without-loosing-image-quality-in-c-or-opencv
	return resized


### Code
# argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--left", help="path to the left image file")
ap.add_argument("-c", "--center", help="path to the center image file")
ap.add_argument("-r", "--right", help="path to the right image file")

args = vars(ap.parse_args())


# read in images, all images in same-level image directory
left = cv2.imread("images/"+args["left"])
center = cv2.imread("images/"+args["center"])
right = cv2.imread("images/"+args["right"])

# scale to same width
width = 600
left_scaled = scaleImage(left, width)
center_scaled = scaleImage(center, width)
right_scaled = scaleImage(right, width)

# display images
cv2.imshow("left", left_scaled)
cv2.imshow("center", center_scaled)
cv2.imshow("right", right_scaled)
cv2.waitKey(0)


