import cv2
import numpy as np
from utils import gaussian,sobel

def myCornerHarris(img, k=0.04, sigma=0.8):
	dx = sobel(img,0)
	dy = sobel(img,1)

	Ix2 = np.full_like(img,0)
	Iy2 = np.full_like(img,0)
	Ixy = np.full_like(img,0)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			Ix2[i][j] = dx[i][j] * dx[i][j]
			Iy2[i][j] = dy[i][j] * dy[i][j]
			Ixy[i][j] = dx[i][j] * dy[i][j]

	A = gaussian(Ix2,sigma)
	B = gaussian(Ixy,sigma)
	C = gaussian(Iy2,sigma)

	pointVal = np.full_like(img,0)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			detM = A[i][j] * C[i][j] - B[i][j] * B[i][j]
			traceM = A[i][j] + C[i][j]
			pointVal[i][j] = detM - k * traceM * traceM

	totalMax = pointVal.max()
	points = []
	for i in range(pointVal.shape[0]):
		for j in range(pointVal.shape[1]):
			maxVal = 0
			for m in [-1,0,1]:
				for n in [-1,0,1]:
					x = max(i+m,0)
					x = min(x,pointVal.shape[0]-1)
					y = max(j+n,0)
					y = min(y,pointVal.shape[1]-1)
					maxVal = max(maxVal,pointVal[x][y])
			if(maxVal == pointVal[i][j] and pointVal[i][j]>0.01*totalMax):
				points.append((i,j))
	return points


# img = cv2.imread('cv_cover.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)

# #dst = cv2.cornerHarris(gray,2,3,0.04)
# corners_points = myCornerHarris(gray,0.04)

# for i in range(len(corners_points)):
# 	img[corners_points[i][0]][corners_points[i][1]] = [0,0,255]

# cv2.imshow('corners',img)
# #cv2.imwrite('corners.png',img)
# if cv2.waitKey(0) & 0xff == 27:
# 	cv2.destroyALLWindows()