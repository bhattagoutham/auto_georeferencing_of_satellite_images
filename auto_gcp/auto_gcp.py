import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math as mt
import sys
from plot import plot_lines

def disp1(r1):
	cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image1', 1200,800)
	cv2.imshow("image1",r1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def disp2(r1, r2):
	cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image1', 600,600)
	cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image2', 600,600)
	
	cv2.imshow("image1",r1)
	cv2.imshow("image2",r2)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def img_segment(fname, k_val):
	
	print('Segmenting '+fname+' image...')

	img = cv2.imread(fname)
	
	Z = img.reshape((-1,3))	
	Z = np.float32(Z)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(Z,k_val,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	
	center = np.uint8(center)
	res = center[label.flatten()]
	
	res= res.reshape((img.shape))
	cv2.imwrite(fname[0:-4]+'_seg.png',res)
	# return res


def blob(fname, lower, upper):
	
	print('Computing blob for '+fname[0:-4]+'_seg.png'+' image......')
	src = cv2.imread(fname[0:-4]+'_seg.png')
	mask = cv2.inRange(src, lower, upper)
	blur = cv2.GaussianBlur(mask,(11,11),0)
	median = cv2.medianBlur(blur,11)
	cv2.imwrite(fname[0:-4]+'_blob.png',median)
	# return median
	

def morph(gray):

	kernel = np.ones((15,15),np.uint8)	#15
	erosion = cv2.erode(gray,kernel,iterations = 3)	#3
	dilation = cv2.dilate(erosion,kernel,iterations = 1)
	return dilation #erosion



def feature_match(im1, im2, sz):

	sift = cv2.xfeatures2d.SIFT_create()
	
	r1 = cv2.resize(im1,None,fx=1/sz, fy=1/sz, interpolation = cv2.INTER_CUBIC)
	r2 = cv2.resize(im2,None,fx=1/sz, fy=1/sz, interpolation = cv2.INTER_CUBIC)

	
	(kp1, des1) = sift.detectAndCompute(r1,None)
	(kp2, des2) = sift.detectAndCompute(r2,None)

	rt1 = cv2.drawKeypoints(r1,kp1, r1, flags=4)
	rt2 = cv2.drawKeypoints(r2,kp2, r2, flags=4)
	
	disp2(rt1, rt2)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	# Apply ratio test
	good = []
	good_plt = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good.append(m)
	        good_plt.append([m])


	MIN_MATCH_COUNT = 10
	
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()
	print(M)

	h,w = r1.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)
	img2 = cv2.polylines(r2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

	img3 = cv2.drawMatches(r1,kp1,r2,kp2,good,None,**draw_params)

	print(r1.shape, r2.shape, len(matches))
	disp1(img3)
	return M




def detect():

	rec_img_path = "red.png"
	raw_img_path = "green.jpg"

	# img_segment(rec_img_path, 5)
	# img_segment(raw_img_path, 5)
	
	rec_im = cv2.imread("red_seg.png")
	raw_im = cv2.imread("green_seg.png")
	
	# # rec red img => color of water (b,g,r) range values
	# lower = np.array([108, 78, 49]) 
	# upper = np.array([110, 80, 51]) 
	# blob(rec_img_path, lower, upper)
	
	# # raw green im => color of water (b,g,r) range values
	# lower = np.array([46, 41, 23]) #np.array([23, 41, 46])
	# upper = np.array([48, 43, 25]) #np.array([25, 43, 48])
	# blob(raw_img_path, lower, upper)
	

	gray1 = cv2.imread(rec_img_path[0:-4]+'_blob.png',0)
	gray2 = cv2.imread(raw_img_path[0:-4]+'_blob.png',0)

	rec_morph = morph(gray1)
	raw_morph = morph(gray2)

	sz = int(mt.pow(2, 3))
	M = feature_match(rec_morph,raw_morph, sz)
	plot_lines(rec_im, raw_im, M)
	# return M

detect()

	


