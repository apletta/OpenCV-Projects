import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

filename = 'pendulums.jpg'

# Convert to hsv
image = mpimg.imread(filename) # original image
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray scale
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # hsv 

# Cannyed edge detection
upper_threshold = 220
lower_threshold = 180
cannyed_image_hsv = cv2.Canny(hsv_image, lower_threshold, upper_threshold)

plt.imshow(image) # original image

plt.figure()
plt.imshow(cannyed_image_hsv)

plt.show()