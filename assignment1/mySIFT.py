import cv2
import numpy as np
import math
import utils
from myHarris import myCornerHarris,sobel

def get_main_direction(img, key_points, sigma_oct=2, radius=4):
    #sobel求图像梯度
    Dx = sobel(img, 0)
    Dy = sobel(img, 1)
    dis = np.zeros(img.shape)
    angles = np.zeros(img.shape)
    gauss_kernel = utils.gaussianKernel2D(1.5*sigma_oct, 2*radius+1)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = Dx[i][j]
            y = Dy[i][j]
            if(x == 0 and y == 0):
                angles[i][j] = 0
                dis[i][j] = 0
            else:
                xy = (x**2+y**2)**0.5
                dis[i][j] = xy
                radian = math.acos(x/xy)
                ag = radian*180/math.pi
                if(y < 0):
                    ag = ag + 180
                if(ag >= 360):
                    ag = 0
                angles[i][j] = ag
    direction_points = []        
    for p in range(len(key_points)):
        x = key_points[p][0]
        y = key_points[p][1]
        #直方图
        h = np.zeros(36)
        for i in range(-radius,radius+1):
            for j in range(-radius,radius+1):
                px = x + i
                py = y + j
                if(i**2+j**2 <= radius**2 and px >= 0 and px < img.shape[0] and py >= 0 and py < img.shape[1]):
                    ag = int(angles[px][py]/10)
                    h[ag] = h[ag] + gauss_kernel[radius+i][radius+j]*dis[px][py]

        #梯度平滑处理
        H = np.zeros(36)
        for i in range(H.shape[0]):
            H[i] = 3*h[i]/8 + (h[(i+35)%36]+h[(i+1)%36])/4 + (h[(i+34)%36]+h[(i+2)%36])/16

        maxDirecct = max(H)
        for i in range(len(H)):
            if(H[i] > maxDirecct*0.8):
                #抛物线插值
                left_bin = (i + 35)%36
                right_bin = (i + 1)%36
                if(H[i] > H[left_bin] and H[i] > H[right_bin]):
                    t_max = (H[left_bin]-H[right_bin]) / (2*(H[left_bin]+H[right_bin]-2*H[i]))
                    direction_points.append((x,y,(i+t_max)*10))
                #direction_points.append((x,y,i*10))
    return direction_points, Dx, Dy, angles

def get_descriptor(img, direction_points, Dx, Dy, angles, sigma_oct=2, d=4):
    half_sampling = int(3*sigma_oct*d/2) + int(3*sigma_oct/2)
    descriptors = []
    points = []
    for p in range(len(direction_points)):
        keyP = direction_points[p]
        b = keyP[0]
        a = keyP[1]
        sinP = math.sin(keyP[2])
        cosP = math.cos(keyP[2])
        descriptor = np.zeros((d,d,8))
        for x in range(-half_sampling,half_sampling+1):
            for y in range(-half_sampling,half_sampling+1):
                if(x+a > 0 and x+a < img.shape[1] and y+b > 0 and y+b < img.shape[0]):
                    xx = x*cosP - y*sinP
                    yy = x*sinP + y*cosP
                    xxx = xx/(3*sigma_oct) + d/2 - 0.5
                    yyy = yy/(3*sigma_oct) + d/2 - 0.5
                    mag = (Dx[y+b][x+a]**2 + Dy[y+b][x+a]**2)**0.5
                    mag = mag*math.exp(-(xx**2+yy**2)/(0.5*d**2))

                    #三线性插值
                    r0 = int(xxx)
                    d_r = xxx - r0
                    c0 = int(yyy)
                    d_c = yyy - c0
                    angle = (angles[int(y+b)][int(x+a)]-keyP[2]+360)%360
                    o0 = int(angle/45)
                    d_o = angle/45 - o0
                    for r in [0,1]:
                        rb = r0 + r
                        if(rb >= 0 and rb < d):
                            vr = 0
                            if(r == 0):
                                vr = mag * (1 - d_r)
                            else:
                                vr = mag * d_r
                            for c in [0,1]:
                                cb = c0 + c
                                if(cb >=0 and cb < d):
                                    vc = 0
                                    if(c == 0):
                                        vc = vr * (1 - d_c)
                                    else:
                                        vc = vr * d_c
                                    for o in [0,1]:
                                        ob = (o0 + o)%8
                                        vo = 0
                                        if(o == 0):
                                            vo = vc * (1 - d_o)
                                        else:
                                            vo = vc * d_o
                                        descriptor[rb][cb][ob] = descriptor[rb][cb][ob] + vo
        
        #归一化
        sqrtSumAll = np.sum(descriptor*descriptor)**0.5
        if(sqrtSumAll != 0):
            descriptor = descriptor / sqrtSumAll
            descriptors.append(list(np.array(descriptor).flatten()))
            #descriptors.append(descriptor)
            points.append([b,a])
    return points, descriptors



def SIFT(img, k=0.04, sigma=0.8):
    img = np.float32(img)
    key_points = myCornerHarris(img, k, sigma)
    direction_points, Dx, Dy, angles = get_main_direction(img,key_points)
    points, descriptors = get_descriptor(img, direction_points, Dx, Dy, angles)
    #print(descriptors)
    #print(points)
    
    return points, descriptors

# img2 = cv2.imread('cv_cover.jpg')
# img = scipy.ndimage.rotate(img2, 5, (1,0))
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# #SIFT(gray)
# kp = myCornerHarris(gray, 0.04, 0.8)
# direction_points, Dx, Dy, angles = get_main_direction(gray,kp)
# for i in range(len(direction_points)):
#     y=direction_points[i][0]
#     x=direction_points[i][1]
#     angle = direction_points[i][2]
#     img[y][x] = [0,0,255]
    
# cv2.imshow("aaa",img)
# cv2.waitKey(0)
