from imutils import paths
import numpy as np 
import argparse
import imutils
import cv2

### Resources
# tutorial -->  https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
# paper    -->  http://matthewalunbrown.com/papers/ijcv2007.pdf


### Code
# argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True, help="path to the output image")
args = vars(ap.parse_args())


# put image paths into list so any in the directory can be read 
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = [] # list of images to be stitched
 
# loop through image list and load images to list
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)

# stitch images
print("[INFO] stitching images...")


if imutils.is_cv3(): # OpenCV 3
	stitcher = cv2.createStitcher()
else: # OpenCV 4
	stitcher = cv2.Stitcher_create() 

(status, stitched) = stitcher.stitch(images) # call method to stitch images

# evaluate status and display image if successfully stitched
if status == 0: # status is 0 for successful operation
	cv2.imwrite(args["output"], stitched) # write to output file
	cv2.imshow("Stitched", stitched) # display stitched image
	cv2.waitKey(0)

else: # status is 1, 2 or 3 depending on error (see documentation)
	print("[INFO] image stitching failed ({})".format(status)) # failure message


### Takeaways
# Easy to use
# Slow, not for real-time
# Can implement additional cropping to make image more rectangular
# For real-time, see https://www.pyimagesearch.com/2016/01/25/real-time-panorama-and-image-stitching-with-opencv/






