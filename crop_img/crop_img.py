import cv2 
import os
from imutils import paths
import random

# list all the images in folder ./dataset
folder = 'none'
img_train = './'+ folder
img_list = list(paths.list_images(img_train))
num_crop = len(img_list)
for numb in range(num_crop):
	img = cv2.imread(img_list[numb])
	high = img.shape[0]
	width = img.shape[1]
	x = int(high/2)+60
	y = int(width/2)+60
	crop_img = img[x:x+48,y:y+48]
	cv2.imwrite("./img/crop_"+folder+"_"+str(823+numb)+".png",crop_img)
	# cv2.imshow("img", img)
	# cv2.imshow("crop", crop_img)
	# cv2.waitKey(0)
