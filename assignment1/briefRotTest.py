import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
from matchPics import matchPics
from helper import plotMatches


#Read the image and convert to grayscale, if necessary
img = cv2.imread('cv_cover.jpg')
res_set = []

for i in range(36):
	#Rotate Image
	rimg = scipy.ndimage.rotate(img, 10*i)
	#Compute features, descriptors and Match features
	matches,locs1,locs2 = matchPics(img,rimg)
	#Update histogram
	res_set.append(len(matches))

print(res_set)

#Display histogram
plt.hist(res_set, bins=range(0,1001,100))
plt.xlabel('count of matches')
plt.ylabel('count')
plt.show()