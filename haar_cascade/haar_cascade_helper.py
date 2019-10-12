import urllib.request
import cv2
import numpy as np 
import os

def store_raw_images():
	# stadiums: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04295881
	# tennis courts: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00483705
	# fence: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03000134
	# field: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n08659446
	# track: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00439826
	neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00439826'
	neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()

	if not os.path.exists('neg5'):
		os.makedirs('neg5')

	pic_num = 8000

	for i in neg_image_urls.split('\n'):
		try:
			print(i)
			urllib.request.urlretrieve(i, "neg5/"+str(pic_num)+'.jpg') # retrieve image and save
			img = cv2.imread("neg5/"+str(pic_num)+'.jpg', cv2.IMREAD_GRAYSCALE) # read in image
			resized_image = cv2.resize(img, (100,100)) # resize to 100x100
			cv2.imwrite("neg5/"+str(pic_num)+'.jpg', resized_image)
			pic_num += 1


		except Exception as e:
			print (str(e))

store_raw_images()