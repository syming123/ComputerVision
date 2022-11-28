import cv2
import numpy as np
import random
import helper
import math


# 奇异值分解求解 Ax = 0
def SVD_Ax0(A):
    U,S,V = np.linalg.svd(A)
    return V[-1,:]


# 八点法
def eight_point(pts1, pts2, M):
    N = len(pts1)
    normalize_pts1 = []
    normalize_pts2 = []
    # 归一化
    T = np.array([[1.0/M[1],0,0],[0,1.0/M[0],0],[0,0,1]])
    for i in range(N):
        p1T = np.array([pts1[i][0],pts1[i][1],1])
        p2T = np.array([pts2[i][0],pts2[i][1],1])
        normalize_pts1.append((T@p1T.transpose()).transpose())
        normalize_pts2.append((T@p2T.transpose()).transpose())

    # 计算基础矩阵F
    A = np.zeros((N,9))
    for i in range(N):
        x1 = normalize_pts1[i][0]
        y1 = normalize_pts1[i][1]
        xx1 = normalize_pts2[i][0]
        yy1 = normalize_pts2[i][1]
        A[i] = np.array([x1*xx1,x1*yy1,x1,y1*xx1,y1*yy1,y1,xx1,yy1,1])
    f_vec = SVD_Ax0(A)
    F = np.array([
        [f_vec[0],f_vec[1],f_vec[2]],
        [f_vec[3],f_vec[4],f_vec[5]],
        [f_vec[6],f_vec[7],f_vec[8]]
    ])

    # 将F的秩设为2
    U,s,V = np.linalg.svd(F)
    s[-1] = 0
    S = np.diag(s)
    F_rank2 = U@S@V

    # Refine F
    F = helper.refineF(F,pts1,pts2)

    # 取消归一化
    F_unnorm = T.transpose()@F_rank2@T

    return F_unnorm


# 利用曼哈顿距离寻找匹配点
def epipolar_correspondences(im1,im2,F,pts1):
    RADIUS = 5
    gray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    pad1 = np.pad(gray1,RADIUS)
    pad2 = np.pad(gray2,RADIUS)
    m = gray2.shape[1]
    n = gray2.shape[0]
    pts2 = []
    for i in range(len(pts1)):
        M1 = pad1[pts1[i][1]:pts1[i][1]+2*RADIUS+1,pts1[i][0]:pts1[i][0]+2*RADIUS+1]
        p1 = np.array([pts1[i][0],pts1[i][1],1])
        line = F@p1.transpose()
        a = line[0]
        b = line[1]
        c = line[2]
        min_dis = 999999
        min_point = (0,0)
        for j in range(m):
            x = j
            y = int(-(1/b)*(x*a+c))
            if(y < 0 or y >= n):
                continue
            M2 = pad2[y:y+2*RADIUS+1,x:x+2*RADIUS+1]
            dis = (np.abs(M1-M2)).sum()
            if(min_dis > dis):
                min_dis = dis
                min_point = (x,y)
        pts2.append(np.array(min_point))
    return pts2


# 计算本质矩阵E
def essential_matrix(F,K1,K2):
    E = K2.transpose()@F@K1
    return E


# 三角化
def triangulate(P1,pts1,P2,pts2):
    pts3d = []
    for i in range(len(pts1)):
        u1 = pts1[i][0]
        v1 = pts1[i][1]
        u2 = pts2[i][0]
        v2 = pts2[i][1]
        A = np.array([
            u1*P1[2,:] - P1[0,:],
            v1*P1[2,:] - P1[1,:],
            u2*P2[2,:] - P2[0,:],
            v2*P2[2,:] - P2[1,:]
        ])
        X = SVD_Ax0(A)
        pts3d.append((X/X[3])[:3])   
    return pts3d


# 寻找正确的P2
def figure_correct_P2(pts3ds):
    correct = 0
    max_corr_pts = 0
    for i in range(len(pts3ds)):
        corr_pts = 0
        for j in range(len(pts3ds[i])):
            if(pts3ds[i][j][2] > 0):
                corr_pts = corr_pts + 1
        if(max_corr_pts < corr_pts):
            max_corr_pts = corr_pts
            correct = i
    return correct


# 利用欧氏距离计算重新投影误差
def compute_error(P1,pts3d,pts1):
    dis = 0
    for i in range(len(pts3d)):
        back_pts = P1@np.array([pts3d[i][0],pts3d[i][1],pts3d[i][2],1])
        dis = dis + np.sqrt((((back_pts/back_pts[2])[:2]-pts1[i])**2).sum())
    return dis/len(pts3d)


# 去除过于偏的点
def select_points(pts,th=10):
    sum = np.zeros(3)
    res = []
    for i in range(len(pts)):
        for j in range(3):
            sum[j] = sum[j] + pts[i][j]
    adv = [sum[0]/len(pts),sum[1]/len(pts),sum[2]/len(pts)]
    for i in range(len(pts)):
        if(abs(adv[0]-pts[i][0])<th and abs(adv[1]-pts[i][1])<th and abs(adv[2]-pts[i][2])<th):
            res.append(pts[i])
    return res
