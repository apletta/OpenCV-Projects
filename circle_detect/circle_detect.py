import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# Load image, clone for output, convert to grayscale
image = cv2.imread(args["image"])

# Rescale image, 
# height, width = image.shape[:2]
# image = cv2.resize(image, (int(width/3), int(height/3)))
# cv2.imshow("image", image)
# cv2.waitKey(0)

output = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect circles in image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

# Ensure at least some circles were found
if circles is not None:
	
	# Convert the (x, y) coordinates and radius of the cirlces to integers
	circles = np.round(circles[0, :]).astype("int")

	# Loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle at circle center
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

	# show the output image: np.hstack([image, output])
	cv2.imshow("image", image)
	cv2.imshow("grayscale", gray)
	cv2.imshow("detection", output)
	cv2.waitKey(0)

	plt.show()
else:
	print("No circles detected.")
	














