import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import cv2
import math
import argparse
import os.path


### Read in image

# User input
file = input("Enter file name here: ")
while (not os.path.isfile(file) or not os.path.exists(file)):
	if(file == "q"):
		quit()
	print("Please enter valid file name. Double check directory location and file extension (ex. use .jpg for JPEG images). ")
	file = input("File: ")
view = input("Enter perspective of jump according to skier left or right: " )
while(view != "left" and view != "right"):
	if(view == "q"):
		quit()
	print("Please enter either \"left\" or \"right\"") 
	view = input("Perspective: ")
large_vert_drop = input("Is there a large vertical drop? Selecting yes will extend the trajectory line. (y/n): ")
while(large_vert_drop != "y" and large_vert_drop != "n"):
	if(view == "q"):
		quit()
	print("Please enter either \"y\" or \"n\" if there is an excessive drop after the jump (ex. cliffs) ") 
	large_vert_drop = input("Large vertical drop? " )
# Argument parsing
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# ap.add_argument("-v", "--view", required = True, help = "View from skier's left or right of jump")
# args = vars(ap.parse_args())

# Load image, clone for output, convert to grayscale --> for args implementation only
#image = cv2.imread(args["image"])
# view = args["view"] 

image = cv2.imread(file)
plt.figure("Jump Trajectory Lite")
plt.title("Drag the dots to the locations specified below:")
plt.xlabel("Red --> Jump Take-off, Blue --> Start of Jump, Green --> Drop Point")
plt.imshow(image)

### Prompt user to place points 1, 2, and h_s
x1 = 0
y1 = 0
x2 = 0
y2 = 0
drop_height = 0

class setLocations:
	def __init__(self, rect, color):
		self.rect = rect
		self.press = None
		self.color = color

	def connect(self):
		'connect to all dots that are being clicked on'
		self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
		self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
		self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

	def on_press(self, event):
		'when button is pressed, check if mouse is over a dot and store data'
		if event.inaxes != self.rect.axes: return

		contains, attrd = self.rect.contains(event)

		if not contains: return
		x0, y0 = self.rect.center
		self.press = x0, y0, event.xdata, event.ydata

	def on_motion(self, event):
		'if mouse is clicked and held over dot, move dot along with mouse'
		if self.press is None: return
		x0, y0, xpress, ypress = self.press
		dx = event.xdata - xpress # distance moved in x direction
		dy = event.ydata - ypress # distance moved in y direction
		self.rect.center = x0+dx, y0+dy

	def on_release(self, event):
		'on release, reset press data and keep object in most recent location'
		self.press = None
		self.rect.figure.canvas.draw()
		
	def disconnect(self):
		'disconnect all stored connection ids'
		self.rect.figure.canvas.mpl_disconnect(self.cidpress)
		self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
		self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

img_height = image.shape[0]
img_width = image.shape[1]
circle1 = plt.Circle((img_width*3/12,img_height/6), img_width/150,color='red')
circle2 = plt.Circle((img_width*4/12, img_height/6), img_width/150,color='blue')
circle3 = plt.Circle((img_width*5/12, img_height/6), img_width/150,color='green')

plt.gcf().gca().add_artist(circle1)
plt.gcf().gca().add_artist(circle2)
plt.gcf().gca().add_artist(circle3)

drs = []
dr = setLocations(circle1, 'red')
dr.connect()
drs.append(dr)
dr = setLocations(circle2, 'blue')
dr.connect()
drs.append(dr)
dr = setLocations(circle3, 'green')
dr.connect()
drs.append(dr)

plt.show()

x1 = circle2.center[0]
y1 = circle2.center[1]
x2 = circle1.center[0]
y2 = circle1.center[1]
drop_height = circle3.center[1]


### Calculations
gravity = 32.2
if(view == "left"):
	jump_distance = x1 - x2
	# print("d_j "+str(jump_distance))
else:
	jump_distance = x2 - x1
	# print("d_j "+str(jump_distance))
jump_height = y1 - y2
# print('h_j '+str(jump_height))
# print('h_d '+str(drop_height))

if(drop_height >= y2):
	print("Drop-in height less than or equal to jump take-off, no launch expected. ")
	plt.figure("Jump Trajectory Lite")
	plt.imshow(image)
	plt.title("Drop-in height less than or equal to jump take-off, no launch expected. ")
	plt.show()
elif(jump_height<0):
	print("Jump take-off lower than bottom of jump. Re-design jump. ")
	plt.figure("Jump Trajectory Lite")
	plt.imshow(image)
	plt.title("Jump take-off lower than bottom of jump. Re-design jump. ")
	plt.show()
else:
	theta = math.atan(jump_height/jump_distance)
	# print("thet "+str(theta))
	jump_velocity = math.sqrt(2*gravity*(y2-drop_height))
	# print("v_j "+str(jump_velocity))
	peak_y = math.pow(jump_velocity*math.sin(theta), 2) / (2*gravity)
	# print("peak_y "+ str(peak_y))
	max_y = jump_height + peak_y
	# print("max_y "+ str(max_y))
	air_time = math.sqrt(2*max_y/gravity) + jump_velocity*math.sin(theta)/gravity
	# print("t "+str(air_time))
	total_distance = jump_velocity*math.cos(theta)*air_time
	# print("d "+str(total_distance))

	# Trajectory
	def x_traj(time, velocity, theta, view):
		if(view == "right"):
			return velocity*math.cos(theta)*time + x2
		else:
			return velocity*math.cos(theta)*time*-1 + x2

	def y_traj(time, velocity, theta, gravity, launch_height):
		return -velocity*math.sin(theta)*time + gravity*np.power(time, 2)/2 + y2

	### Display data
	print()
	print("Distance in air: "+str(total_distance)[:5]+" ft")
	print("Air time: "+str(air_time)[:4]+" s")
	print("Max height: "+str(max_y)[:4]+" ft")
	print()

	### Overlay image with trajectory
	if(large_vert_drop == 'n'):
		time = np.arange(0., air_time, .1)
	else:
		time = np.arange(0., air_time*2, .1)
		
	plt.figure("Jump Trajectory Lite")
	plt.imshow(image)
	plt.gcf().gca().plot(x_traj(time, jump_velocity, theta, view), y_traj(time, jump_velocity, theta, gravity, jump_height) , 'r--')
	plt.show()























