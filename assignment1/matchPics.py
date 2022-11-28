import numpy as np
import cv2
import skimage.color
import scipy.ndimage
from helper import briefMatch
from helper import computeBrief
from helper import fast_corner_detection
from helper import plotMatches

def matchPics(I1, I2):
	#I1, I2 : Images to match

	#Convert Images to GrayScale
	gray1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	corner_locs1 = fast_corner_detection(gray1)
	corner_locs2 = fast_corner_detection(gray2)
	
	#Obtain descriptors for the computed feature locations
	desc1,locs1 = computeBrief(gray1,corner_locs1)
	desc2,locs2 = computeBrief(gray2,corner_locs2)

	#Match features using the descriptors
	matches = briefMatch(desc1,desc2)

	return matches, locs1, locs2



img1 = cv2.imread('cv_cover.jpg')
img2 = scipy.ndimage.rotate(img1, 80, (1,0))

matches,locs1,locs2 = matchPics(img1,img2)
plotMatches(img1,img2,matches,locs1,locs2)

if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyALLWindows()