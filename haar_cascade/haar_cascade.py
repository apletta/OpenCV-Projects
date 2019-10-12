import urllib.request
import cv2
import numpy as np 
import os

def store_raw_images():
	# stadiums: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04295881
	# tennis courts: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00483705
	# fence: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03000134
	# field: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n08659446
	neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n08659446'
	neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()

	if not os.path.exists('neg2'):
		os.makedirs('neg2')

	pic_num = 10000

	for i in neg_image_urls.split('\n'):
		try:
			print(i)
			urllib.request.urlretrieve(i, "neg2/"+str(pic_num)+'.jpg') # retrieve image and save
			img = cv2.imread("neg2/"+str(pic_num)+'.jpg', cv2.IMREAD_GRAYSCALE) # read in image
			resized_image = cv2.resize(img, (100,100)) # resize to 100x100
			cv2.imwrite("neg2/"+str(pic_num)+'.jpg', resized_image)
			pic_num += 1


		except Exception as e:
			print (str(e))

# store_raw_images() # run this to pull and save image files to folder 

def find_uglies():
	for file_type in ['neg']:
		for img in os.listdir(file_type):
			for ugly in os.listdir('uglies'):
				try:
					current_image_path = str(file_type)+'/'+str(img)
					ugly = cv2.imread('uglies/'+str(ugly))
					question = cv2.imread(current_image_path)

					if ugly.shape == question.shape	and not(np.bitwise_xor(ugly,question).any()):
						print('ugly image')
						print(current_image_path)
						os.remove(current_image_path)

				except Exception as e:
					print(str(e))

# find_uglies() # run to find and remove blank images ("uglies")

def create_pos_n_neg():
	for file_type in ['neg']:

		for image in os.listdir(file_type):
			if file_type == 'neg':
				line = file_type+'/'+image+'\n'
				with open('bg.txt', 'a') as f:
					f.write(line)

			elif file_type == 'pos':
				line = file_type+'/'+image+'1 0 0 50 50\n'
				with open('info.dat', 'a') as f:
					f.write(line)

# create_pos_n_neg() # move negative files to a text file






