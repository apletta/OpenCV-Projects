import numpy as np 
import argparse
import cv2

### resources
#https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
#https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/

# argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-r", "--radius", type=int, help="radius of Gaussian blur; must be odd")
ap.add_argument("-t", "--threshold", type=int, help="threshold value for detecting brightness")
args = vars(ap.parse_args())

maxThresh = args["threshold"] # threshold for bright spot detection

### thresholds, w radius=51 (smaller radius is more sensitive because of Gaussian blur being applied)
# sun-glare 250
# driving 245 (not really any glare..)
# shock-wave 220
# bromo 240 (no glare, don't want false positives on dust)

### --> issues to solve:
# if camera auto-focuses/brightens, could affect expected threshold values
# resolution of image/frame affects radius size, so therefore blurring effect, and therefore threshold values

# read in video file, pass in 0 to read from connected camera
cap = cv2.VideoCapture(args["video"])

# check video/camera properly opened
if (cap.isOpened()==False):
	print("Error opening video stream or file")

else:
	# read until video/stream is complete
	while(cap.isOpened()):

		# capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:

			# load image and convert to grayscale
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			#cv2.imshow("hsv", hsv)
			#cv2.waitKey(0)

			# apply Gaussian blur to image before finding bright region
			gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)

			(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
			image = frame.copy()
			
			# only detect above maxThresh
			if maxVal >= maxThresh:
				cv2.circle(image, maxLoc, args["radius"], (255, 0, 0), 2)
				#cv2.circle(gray, maxLoc, args["radius"], (255, 0, 0), 2)


			print("minVal", minVal, "|", "maxVal", maxVal)

			# display resulting frame 
			cv2.imshow("Frame", image)
			#cv2.imshow("Grayscaled blur", gray)


			# press q on keyboard to exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break


		# break loop
		else:
			break

# when done, hold until user hits key
cv2.waitKey(0)

# release video capture
cap.release()

# close all frames
cv2.destroyAllWindows()



