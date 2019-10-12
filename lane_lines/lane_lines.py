import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import cv2
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML
 
### METHODS ###

def region_of_interest(img, vertices):
	# Define a blank matrix that matches the image height/width
	mask = np.zeros_like(img)

	# Retrieve the number of color channels of the image
	

	# Create a match color with the same color channel counts

	# for color
	#channel_count = img.shape[2]
	# match_mask_color = (255,) * channel_count

	# for grayscale
	match_mask_color = 255  # grayscale match_mask

	# Fill inside the polygon
	cv2.fillPoly(mask, vertices, match_mask_color)

	# Returning the image only where mask pixels match
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def region_of_interest_original(img, vertices):
	# Define a blank matrix that matches the image height/width
	mask = np.zeros_like(img)

	# Retrieve the number of color channels of the image
	

	# Create a match color with the same color channel counts

	# for color
	channel_count = img.shape[2]
	match_mask_color = (255,) * channel_count

	# # for grayscale
	# match_mask_color = 255  # grayscale match_mask

	# Fill inside the polygon
	cv2.fillPoly(mask, vertices, match_mask_color)

	# Returning the image only where mask pixels match
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

# Hough Lines Detection
def make_lines(img):
	lines = cv2.HoughLinesP(img, 
						rho=6, 
						theta=np.pi/60, 
						threshold=70, 
						lines=np.array([]), 
						minLineLength=40,
						maxLineGap=30)
	return lines

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
	# If there are no lines, exit
	if lines is None:
		return

	# Make a copy of the original image
	img = np.copy(img)

	# Create a blank image that matches the original in size
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

	# Loop over all lines and draw them on the blank image
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

	# Merge the image with the lines onto the original
	img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

	# Return the modified image
	return img

def pipeline(image):
	"""
	Image processing pipeline which outputs an image with annotated lane lines
	"""

	# Convert image to grayscale
	gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# Call Canny Edge Detection
	upper_threshold = 80
	lower_threshold = 80
	cannyed_image = cv2.Canny(gray_image, lower_threshold, upper_threshold)

	# Defining region of interest and creating overlayed shape
	height = image.shape[0]
	width = image.shape[1]
	left_bound = 0
	right_bound = width 
	top_bound = height / 2.8
	bottom_bound = height 
	center = width / 2
	region_of_interest_vertices = [(left_bound, bottom_bound), (center, top_bound), (right_bound, bottom_bound)]
	cropped_image_cannyed = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))
	cropped_image_original = region_of_interest_original(image, np.array([region_of_interest_vertices], np.int32))

	# Make lines
	lines = make_lines(cropped_image_cannyed)

	# Build left and right lanes
	left_line_x = []
	left_line_y = []
	right_line_x = []
	right_line_y = []

	for line in lines:
		for x1, y1, x2, y2 in line:
			slope = (y2 - y1) / (x2 - x1) # Calculating slope
			if math.fabs(slope) < 0.5: # Eliminate near horizontal lines
				continue
			if slope > 0: # If slope negative, make left lane
				left_line_x.extend([x1, x2])
				left_line_y.extend([y1, y2])
			elif slope < 0: # Otherwise slope is positive, make right lane
				right_line_x.extend([x1, x2])
				right_line_y.extend([y1, y2])
	# Define vertical bounds
	min_y = int(height / 2) # Just below horizon
	max_y = int(height / 1.15) # Just above dash

	# Apply linear fits
	poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
	left_x_start = int(poly_left(max_y)) 
	left_x_end = int(poly_left(min_y)) 

	poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
	right_x_start = int(poly_right(max_y))
	right_x_end = int(poly_right(min_y))

	# Overlay lines on original image
	line_image = draw_lines(
		image, [[
					[left_x_start, max_y, left_x_end, min_y],
					[right_x_start, max_y, right_x_end, min_y],
				]], 
				thickness=5) # Fitted lines

	return line_image


# ### OPERATIONS - w.o pipeline ###

# # Reading in an image
# image = mpimg.imread('drivingPic_morn.jpg')

# # Convert image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# # Call Canny Edge Detection
# cannyed_image = cv2.Canny(gray_image, 80, 140)

# # Defining region of interest and creating overlayed shape
# height = image.shape[0]
# width = image.shape[1]
# left_bound = 0
# right_bound = width 
# top_bound = height / 2.8
# bottom_bound = height 
# center = width / 2
# region_of_interest_vertices = [(left_bound, bottom_bound), (center, top_bound), (right_bound, bottom_bound)]
# cropped_image_cannyed = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))
# cropped_image_original = region_of_interest_original(image, np.array([region_of_interest_vertices], np.int32))

# # Make lines
# lines = make_lines(cropped_image_cannyed)

# # Build left and right lanes
# left_line_x = []
# left_line_y = []
# right_line_x = []
# right_line_y = []

# for line in lines:
# 	for x1, y1, x2, y2 in line:
# 		slope = (y2 - y1) / (x2 - x1) # Calculating slope
# 		if math.fabs(slope) < 0.5: # Eliminate near horizontal lines
# 			continue
# 		if slope >= 0: # If slope negative, make left lane
# 			left_line_x.extend([x1, x2])
# 			left_line_y.extend([y1, y2])
# 		else: # Otherwise slope is positive, make right lane
# 			right_line_x.extend([x1, x2])
# 			right_line_y.extend([y1, y2])

# # Define vertical bounds
# min_y = int(height / 2.2) # Just below horizon
# max_y = int(height / 1.15) # Just above dash

# # Apply linear fits
# poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
# left_x_start = int(poly_left(max_y)) 
# left_x_end = int(poly_left(min_y)) 

# poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
# right_x_start = int(poly_right(max_y))
# right_x_end = int(poly_right(min_y))

# # Overlay lines on original image
# line_image_hough = draw_lines(image, lines) # Hough lines
# line_image_fitted = draw_lines(
# 	image, [[
# 				[left_x_start, max_y, left_x_end, min_y],
# 				[right_x_start, max_y, right_x_end, min_y],
# 			]], 
# 			thickness=20) # Fitted lines


# ### DISPLAY - w/o pipeline ####

# print('This image is:', type(image), 'with dimensions:', image.shape) # Stats and plotting the image

# plt.imshow(image) # Show original image

# plt.figure()
# plt.imshow(cropped_image_original) # Show original cropped image

# plt.figure()
# plt.imshow(cropped_image_cannyed) # Show cannyed cropped image

# plt.figure()
# plt.imshow(cannyed_image) # Show canny lines

# plt.figure()
# plt.imshow(line_image_hough) # Show overlayed Hough lines

# plt.figure()
# plt.imshow(line_image_fitted) # Show overlayed fitted lines

# plt.show()

### USING PIPELINE ###

# # Print image
# plt.figure()
# plt.imshow(pipeline(mpimg.imread('drivingPic_morn.jpg'))) # Show overlayed fitted lines
# plt.show()

# Read in from video
video = 'drivingVid_morn.mov'
output = video[:-4] + '_output.mp4'
clip1 = VideoFileClip(video)
clip = clip1.fl_image(pipeline)
clip.write_videofile(output, audio=False)






