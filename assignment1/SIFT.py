import cv2
import numpy as np
import math
import utils
import warnings
warnings.filterwarnings("ignore")

#高斯差分金字塔
def get_pyramid(image, sigma0=1.52):
	octave = int(math.log2(min(image.shape[0],image.shape[1])) - 3)
	intvls = 3
	pyramid = []
	for i in range(octave):
		pyramid.append([])
		for j in range(intvls+3):
			if(i == 0 and j == 0):
				pyramid[i].append(image.copy())
			elif(j == 0):
				pyramid[i].append(cv2.pyrDown(pyramid[i-1][intvls]))
			else:
				sigma = sigma0*(2**(i+j/intvls))
				pyramid[i].append(utils.gaussian(pyramid[i][0],sigma))
	DoG = []
	for i in range(octave):
		DoG.append([])
		for j in range(intvls+2):
			dif_image = pyramid[i][j+1] - pyramid[i][j]
			DoG[i].append(dif_image)
	return pyramid, DoG

#Taylor精确定位，Hessian矩阵去除边缘效应
def exact_location(DoG, octave, itv, x, y, T, sigma, n=3, iteration=5, border=5, hessian_r=10):
	point = []

	img_scale = 1.0 / 255
	deriv_scale = img_scale * 0.5
	second_deriv_scale = img_scale
	cross_deriv_scale = img_scale * 0.25

	img = DoG[octave][itv]
	for i in range(iteration):
		if itv < 1 or itv > n or y < border or y >= img.shape[1] - border or x < border or x >= img.shape[0] - border:
			return None, None, None, None

		img = DoG[octave][itv]
		img_prev = DoG[octave][itv - 1]
		img_next = DoG[octave][itv + 1]

		dD = [(img[x, y + 1] - img[x, y - 1]) * deriv_scale,
			  (img[x + 1, y] - img[x - 1, y]) * deriv_scale,
			  (img_next[x, y] - img_prev[x, y]) * deriv_scale]

		v2 = img[x, y] * 2
		dx2 = (img[x, y + 1] + img[x, y - 1] - v2) * second_deriv_scale
		dy2 = (img[x + 1, y] + img[x - 1, y] - v2) * second_deriv_scale
		ds2 = (img_next[x, y] + img_prev[x, y] - v2) * second_deriv_scale
		dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[x - 1, y - 1]) * cross_deriv_scale
		dxs = (img_next[x, y + 1] - img_next[x, y - 1] - img_prev[x, y + 1] + img_prev[x, y - 1]) * cross_deriv_scale
		dys = (img_next[x + 1, y] - img_next[x - 1, y] - img_prev[x + 1, y] + img_prev[x - 1, y]) * cross_deriv_scale

		H = [[dx2, dxy, dxs],
			 [dxy, dy2, dys],
			 [dxs, dys, ds2]]

		X = np.matmul(np.linalg.pinv(np.array(H)), np.array(dD))

		xi = -X[2]
		xr = -X[1]
		xc = -X[0]

		if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
			break

		y += int(np.round(xc))
		x += int(np.round(xr))
		itv += int(np.round(xi))

	if (i >= iteration):
		return None, x, y, itv
	if (itv < 1 or itv > n or y < border or y >= img.shape[1] - border or x < border or x >= img.shape[0] - border):
		return None, None, None, None

	t = (np.array(dD)).dot(np.array([xc, xr, xi]))

	contr = img[x, y] * img_scale + t * 0.5
	
	if (np.abs(contr) * n < T):
		return None, x, y, itv

	trH = dx2 + dy2
	detH = dx2 * dy2 - dxy * dxy
	if(trH**2 / detH >= (hessian_r+1)**2 / hessian_r):
		return None, x, y, itv

	point.append((x + xr) * (1 << octave))
	point.append((y + xc) * (1 << octave))
	point.append(octave + (itv << 8) + (int(np.round((xi + 0.5)) * 255) << 16))
	point.append(sigma * np.power(2.0, (itv + xi) / n) * (1 << octave) * 2)

	return point, x, y, itv


def get_direction(img, r, c, radius, sigma, BinNum):
	expf_scale = -1.0 / (2.0 * sigma * sigma)

	X = []
	Y = []
	W = []
	temphist = []

	for i in range(BinNum):
		temphist.append(0.0)

	k = 0
	for i in range(-radius, radius + 1):
		y = r + i
		if(y <= 0 or y >= img.shape[0] - 1):
			continue
		for j in range(-radius, radius + 1):
			x = c + j
			if(x <= 0 or x >= img.shape[1] - 1):
				continue

			dx = (img[y, x + 1] - img[y, x - 1])
			dy = (img[y - 1, x] - img[y + 1, x])
			X.append(dx)
			Y.append(dy)
			W.append((i * i + j * j) * expf_scale)
			k += 1
	length = k

	W = np.exp(np.array(W))
	Y = np.array(Y)
	X = np.array(X)
	Ori = np.arctan2(Y, X) * 180 / np.pi
	Mag = (X ** 2 + Y ** 2) ** 0.5

	for k in range(length):
		bin = int(np.round((BinNum / 360.0) * Ori[k]))
		if bin >= BinNum:
			bin -= BinNum
		if bin < 0:
			bin += BinNum
		temphist[bin] += W[k] * Mag[k]

	temp = [temphist[BinNum - 1], temphist[BinNum - 2], temphist[0], temphist[1]]
	temphist.insert(0, temp[0])
	temphist.insert(0, temp[1])
	temphist.insert(len(temphist), temp[2])
	temphist.insert(len(temphist), temp[3])

	hist = []
	for i in range(BinNum):
		hist.append(
			(temphist[i] + temphist[i + 4]) * (1.0 / 16.0) + (temphist[i + 1] + temphist[i + 3]) * (4.0 / 16.0) +
			temphist[i + 2] * (6.0 / 16.0))
	maxval = max(hist)

	return maxval, hist


def get_key_point(DoG, sigma0, pyramid, n, BinNum=36, T=0.04):
	sigma0 = 1.52
	SIFT_ORI_RADIUS = 3 * sigma0
	SIFT_ORI_PEAK_RATIO = 0.8

	points = []
	for i in range(len(DoG)):
		for j in range(1, len(DoG[0]) - 1):
			threshold = 0.5 * T / (n * 255)
			img_prev = DoG[i][j - 1]
			img = DoG[i][j]
			img_next = DoG[i][j + 1]
			for m in range(img.shape[0]):
				for n in range(img.shape[1]):
					#判断是否为极值点
					val = img[m][n]			
					maxVal = val
					minVal = val
					for dx in [-1,0,1]:
						px = max(0,m+dx)
						px = min(px,img.shape[0]-1)
						for dy in [-1,0,1]:
							py = max(0,n+dy)
							py = min(py,img.shape[1]-1)
							maxVal = max(maxVal,img_prev[px][py],img[px][py],img_next[px][py])
							minVal = min(minVal,img_prev[px][py],img[px][py],img_next[px][py])
					if(np.abs(val) > threshold and ((val > 0 and val == maxVal) or (val < 0 and val == minVal))):
						point, x, y, layer = exact_location(DoG, i, j, m, n, T, sigma0)
						if(point == None):
							continue
						scl_octv = point[-1] * 0.5 / (1 << i)
						omax, hist = get_direction(pyramid[i][layer], x, y, int(np.round(SIFT_ORI_RADIUS * scl_octv)),
													  sigma0 * scl_octv, BinNum)
						mag_thr = omax * SIFT_ORI_PEAK_RATIO
						for k in range(BinNum):
							if(k > 0):
								l = k - 1
							else:
								l = BinNum - 1
							if(k < BinNum - 1):
								r2 = k + 1
							else:
								r2 = 0
							if(hist[k] > hist[l] and hist[k] > hist[r2] and hist[k] >= mag_thr):
								bin = k + 0.5 * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[k] + hist[r2])
								if (bin < 0):
									bin = BinNum + bin
								else:
									if(bin >= BinNum):
										bin = bin - BinNum
								temp = point[:]
								temp.append((360.0 / BinNum) * bin)
								points.append(temp)
	return points


def calculate_descriptor(img, ptf, ori, scl, d, n, SIFT_DESCR_SCL_FCTR=3.0, SIFT_DESCR_MAG_THR=0.2,
					   SIFT_INT_DESCR_FCTR=512.0, FLT_EPSILON=1.19209290E-07):
	dst = []
	pt = [int(np.round(ptf[0])), int(np.round(ptf[1]))]
	cos_t = np.cos(ori * (np.pi / 180))
	sin_t = np.sin(ori * (np.pi / 180))
	bins_per_rad = n / 360.0
	exp_scale = -1.0 / (d * d * 0.5)
	hist_width = SIFT_DESCR_SCL_FCTR * scl

	radius = int(np.round(hist_width * 1.4142135623730951 * (d + 1) * 0.5))
	cos_t /= hist_width
	sin_t /= hist_width
	rows = img.shape[0]
	cols = img.shape[1]

	hist = [0.0] * ((d + 2) * (d + 2) * (n + 2))
	X = []
	Y = []
	RBin = []
	CBin = []
	W = []

	k = 0
	for i in range(-radius, radius + 1):
		for j in range(-radius, radius + 1):
			c_rot = j * cos_t - i * sin_t
			r_rot = j * sin_t + i * cos_t
			rbin = r_rot + d // 2 - 0.5
			cbin = c_rot + d // 2 - 0.5
			r = pt[1] + i
			c = pt[0] + j
			if(rbin > -1 and rbin < d and cbin > -1 and cbin < d and r > 0 and r < rows - 1 and c > 0 and c < cols - 1):
				dx = (img[r, c + 1] - img[r, c - 1])
				dy = (img[r - 1, c] - img[r + 1, c])
				X.append(dx)
				Y.append(dy)
				RBin.append(rbin)
				CBin.append(cbin)
				W.append((c_rot * c_rot + r_rot * r_rot) * exp_scale)
				k += 1

	length = k
	Y = np.array(Y)
	X = np.array(X)
	Ori = np.arctan2(Y, X) * 180 / np.pi
	Mag = (X ** 2 + Y ** 2) ** 0.5
	W = np.exp(np.array(W))

	for k in range(length):
		rbin = RBin[k]
		cbin = CBin[k]
		obin = (Ori[k] - ori) * bins_per_rad
		mag = Mag[k] * W[k]

		r0 = int(rbin)
		c0 = int(cbin)
		o0 = int(obin)
		rbin -= r0
		cbin -= c0
		obin -= o0

		if (o0 < 0):
			o0 += n
		if (o0 >= n):
			o0 -= n

		v_r1 = mag * rbin
		v_r0 = mag - v_r1

		v_rc11 = v_r1 * cbin
		v_rc10 = v_r1 - v_rc11

		v_rc01 = v_r0 * cbin
		v_rc00 = v_r0 - v_rc01

		v_rco111 = v_rc11 * obin
		v_rco110 = v_rc11 - v_rco111

		v_rco101 = v_rc10 * obin
		v_rco100 = v_rc10 - v_rco101

		v_rco011 = v_rc01 * obin
		v_rco010 = v_rc01 - v_rco011

		v_rco001 = v_rc00 * obin
		v_rco000 = v_rc00 - v_rco001

		idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0
		hist[idx] += v_rco000
		hist[idx + 1] += v_rco001
		hist[idx + (n + 2)] += v_rco010
		hist[idx + (n + 3)] += v_rco011
		hist[idx + (d + 2) * (n + 2)] += v_rco100
		hist[idx + (d + 2) * (n + 2) + 1] += v_rco101
		hist[idx + (d + 3) * (n + 2)] += v_rco110
		hist[idx + (d + 3) * (n + 2) + 1] += v_rco111

	for i in range(d):
		for j in range(d):
			idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2)
			hist[idx] += hist[idx + n]
			hist[idx + 1] += hist[idx + n + 1]
			for k in range(n):
				dst.append(hist[idx + k])

	nrm2 = 0
	length = d * d * n
	for k in range(length):
		nrm2 += dst[k] * dst[k]
	thr = np.sqrt(nrm2) * SIFT_DESCR_MAG_THR

	nrm2 = 0
	for i in range(length):
		val = min(dst[i], thr)
		dst[i] = val
		nrm2 += val * val
	nrm2 = SIFT_INT_DESCR_FCTR / max(np.sqrt(nrm2), FLT_EPSILON)
	for k in range(length):
		dst[k] = min(max(dst[k] * nrm2, 0), 255)
	return dst


def get_descriptor(gpyr, keypoints, SIFT_DESCR_WIDTH=4, SIFT_DESCR_HIST_BINS=8):
	d = SIFT_DESCR_WIDTH
	n = SIFT_DESCR_HIST_BINS
	descriptor = []
	for i in range(len(keypoints)):
		kpt = keypoints[i]
		o = kpt[2] & 255
		s = (kpt[2] >> 8) & 255
		scale = 1.0 / (1 << o)
		size = kpt[3] * scale
		ptf = [kpt[1] * scale, kpt[0] * scale]
		img = gpyr[o][s]
		descriptor.append(calculate_descriptor(img, ptf, kpt[-1], size * 0.5, d, n))
	return descriptor


def SIFT(img, sigma0=1.52, n=3):
	img = np.float32(img)
	GuassianPyramid,DoG = get_pyramid(img,sigma0)
	KeyPoints = get_key_point(DoG, sigma0, GuassianPyramid, n)
	discriptors = get_descriptor(GuassianPyramid, KeyPoints)
	return KeyPoints, discriptors
