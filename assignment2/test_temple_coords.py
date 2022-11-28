import cv2
import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import functions as func
import helper

# 1. Load the two temple images and the points from data/some_corresp.npz
img1 = cv2.imread("data/im1.png")
img2 = cv2.imread("data/im2.png")
some_corresp = np.load("data/some_corresp.npz")
some_pts1 = some_corresp['pts1']
some_pts2 = some_corresp['pts2']
M = img1.shape


# 2. Run eight_point to compute F
F = func.eight_point(some_pts1,some_pts2,M)
# helper.displayEpipolarF(img1,img2,F)


# 3. Load the points in image 1 contained in data/temple_coords.npz 
# and run your epipolar_correspondences on them to get the corresponding points in image 2 
temple_coords = np.load("data/temple_coords.npz")
all_pts1 = temple_coords['pts1']
# helper.epipolarMatchGUI(img1,img2,F)
all_pts2 = func.epipolar_correspondences(img1,img2,F,all_pts1)


# 4. Load data/intrinsics.npz and compute the essential matrix E
intrinsics = np.load("data/intrinsics.npz")
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E = func.essential_matrix(F,K1,K2)
# print(E)


# 5. Compute the camera projection matrix P1
R1 = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0]
])
P1 = K1@R1
# print(P1)


# 6. Use camera2 to get 4 camera projection matrices P2
R2s = helper.camera2(E)
P2s = []
for i in range(R2s.shape[2]):
    R2i = R2s[:,:,i]
    P2s.append(K2@R2i)


# 7. Run triangulate using the projection matrices
pts3ds = []
for i in range(len(P2s)):
    pts3ds.append(func.triangulate(P1,some_pts1,P2s[i],some_pts2))



# 8. Figure out the correct P2
correct = func.figure_correct_P2(pts3ds)
P2 = P2s[correct]
# err = func.compute_error(P1,pts3ds[correct],some_pts1)
# print(err)


# 9. Scatter plot the correct 3D points
all_pts3d = func.triangulate(P1,all_pts1,P2,all_pts2)
final_points3d = np.array(func.select_points(all_pts3d))
ax = plt.figure().add_subplot(projection = '3d')
ax.scatter(final_points3d[:,0], final_points3d[:,1], zs=final_points3d[:,2], zdir="z", c="#00DDAA", marker="o")
ax.set(xlabel="X", ylabel="Y", zlabel="Z")
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
R1 = P1[:,:3]
T1 = P1[:,3]
R2 = P2[:,:3]
T2 = P2[:,3]
np.savez('data/extrinsics.npz', R1=R1, T1=T1, R2=R2, T2=T2)
