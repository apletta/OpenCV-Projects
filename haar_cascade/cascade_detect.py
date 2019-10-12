import cv2
import numpy as np 

ball_cascade = cv2.CascadeClassifier('data/cascade_12stages.xml')

image = cv2.imread("ball_test2.jpg")

# while True:

# 	ret, img = image.read()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
balls = ball_cascade.detectMultiScale(gray, 50, 50)

numBalls = 0;

for(x, y, w, h) in balls:
	numBalls += 1
	cv2.rectangle(image, (x,y), (x+w, y+h), (255, 255, 0), 2)
if numBalls == 0:
	h = image.shape[0]
	w = image.shape[1]
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image, 'no balls', (int(w/2), int(h/2)), font, 0.5, (0,255,255), 2, cv2.LINE_AA)

cv2.imshow('img', image)
cv2.waitKey(0)