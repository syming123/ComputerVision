import cv2
import numpy as np
import math


def convolution(img, kernel):
    size = len(kernel)
    padding = size // 2
    res = np.zeros((img.shape[0] + padding * 2, img.shape[1] + padding * 2), dtype=np.float32)
    res[padding: padding + img.shape[0], padding: padding + img.shape[1]] = img.copy().astype(np.float32)
    temp = res.copy()
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[padding + i, padding + j] = np.sum(kernel * temp[i: i + size, j: j + size])
    
    return res[padding: padding + img.shape[0], padding: padding + img.shape[1]]


#sobel type = 0为水平方向，type = 1为竖直方向
def sobel(img, type):
	kernels = [
		[[1,0,-1],[2,0,-2],[1,0,-1]],
		[[1,2,1],[0,0,0],[-1,-2,-1]]
	]
	return convolution(img,kernels[type])


def gaussianKernel1D(sigma):
    size = int(int(6*sigma-1)/2)*2+1
    kernel = np.zeros((size))
    center = int(size/2)
    sum = 0
    for i in range(size):
        x2 = 1.0*(i-center)**2
        t = math.exp(-x2/(sigma**2*2))
        kernel[i] = t/((2*np.pi)**0.5*sigma)
        sum = sum + kernel[i]
    kernel = kernel / sum
    return kernel


def gaussianKernel2D(sigma, size):
    kernel = np.zeros((size, size))
    center = int(size/2)
    sum = 0
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            t = math.exp(-(x**2+y**2)/(2*sigma**2))
            kernel[i][j] = t/(2*np.pi*sigma**2)
            sum += kernel[i, j]
    kernel = kernel/sum    
    return kernel



def gaussian(img, sigma = 0.8):
    size = int(int(6*sigma-1)/2)*2+1
    kernel = gaussianKernel1D(sigma)

    padding = size // 2
    res = np.zeros((img.shape[0] + padding * 2, img.shape[1] + padding * 2), dtype=np.float32)
    res[padding: padding + img.shape[0], padding: padding + img.shape[1]] = img.copy().astype(np.float32)
    temp = res.copy()
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[padding + i, padding + j] = np.sum(kernel * temp[i + padding, j: j + size])
    temp = res.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[padding + i, padding + j] = np.sum(kernel * temp[i: i + size, j + padding])   
    return res[padding: padding + img.shape[0], padding: padding + img.shape[1]]
