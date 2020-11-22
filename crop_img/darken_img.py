import cv2 
import os
from imutils import paths
import random
import numpy as np

# list all the images in folder ./dataset
def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha*img + beta, dtype=int)   # cast pixel values to int
    img_new[img_new>255] = 255
    img_new[img_new<0] = 0
    return img_new

def Excution(img_list,brightness = False, darken = False):
	num_img= len(img_list)
	if(brightness == True):
		alpha = 1.0			#brightness: 1.0 	darken: 0.5
		beta = 35			#brightness: 35		darken: 10
		for numb in range(num_img):
			img = cv2.imread(img_list[numb])
			img_new = change_brightness(img, alpha, beta)	
			cv2.imwrite("./img/brightness_"+folder+"_"+str(numb)+".png",img_new)

	if(darken == True):
		alpha = 0.5			#brightness: 1.0 	darken: 0.5
		beta = 10			#brightness: 35		darken: 10
		for numb in range(num_img):
			img = cv2.imread(img_list[numb])
			img_new = change_brightness(img, alpha, beta)	
			cv2.imwrite("./img/darken_"+folder+"_"+str(numb)+".png",img_new)


if __name__ == "__main__":  
	folder = 'left'
	img_train = './'+ folder
	img_list = list(paths.list_images(img_train))
	img_list = sorted(img_list, key=None, reverse=False)
	Excution(img_list, True, True)
	
	
