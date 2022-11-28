import cv2
import numpy as np
import SIFT
import mySIFT
import scipy.ndimage

def match(img1, img2, ratio=0.5):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    keyPoints1, discriptors1 = mySIFT.SIFT(gray1)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    keyPoints2, discriptors2 = mySIFT.SIFT(gray2)
    matches = []
    for i in range(len(keyPoints1)):
        discript1 = discriptors1[i]
        minVal = 999999
        secondMinVal = 9999999
        index = -1
        for j in range(len(keyPoints2)):
            discript2 = discriptors2[j]
            discript = np.array(discript1) - np.array(discript2)
            dis = discript.dot(discript)
            if(dis < minVal):
                secondMinVal = minVal
                minVal = dis
                index = j
        if(minVal/secondMinVal < ratio):
            matches.append((i,index))
    return keyPoints1,keyPoints2,matches

def rotate_match(img, degree):
    img2 = scipy.ndimage.rotate(img, degree, (1,0))
    img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    kp1,kp2,m = match(img,img2,0.3)

    for i in range(len(kp2)):
        kp2[i][1] = kp2[i][1] + img.shape[1]
    img12 = np.concatenate([img, img2], axis=1)
    for i in range(len(kp1)):
        x = int(kp1[i][0])
        y = int(kp1[i][1])
        img12[x][y] = [0,0,255]
    for i in range(len(kp2)):
        x = int(kp2[i][0])
        y = int(kp2[i][1])
        img12[x][y] = [0,0,255]
    for i in range(len(m)):
        mc = m[i]
        x1 = int(kp1[mc[0]][0])
        y1 = int(kp1[mc[0]][1])
        x2 = int(kp2[mc[1]][0])
        y2 = int(kp2[mc[1]][1])
        cv2.line(img12,(y1,x1),(y2,x2),(0,0,255))
    return img12



img = cv2.imread('cv_cover.jpg')
match_img = rotate_match(img, 30)
cv2.imshow('match',match_img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyALLWindows()

