import matplotlib.pyplot as plt
from matplotlib.lines import Line2D      
import matplotlib.image as mpimg
import numpy as np
from numpy.linalg import inv
import cv2
import sys
import math as mt

def disp(r1):

	cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image1', 1200,800)
	cv2.imshow("image1",r1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



def generate_lat_long(r, c, ax1, ax2,M, grid_size):

	# bl => bottom left
	br = [6274, 7241]
	tr = [7833, 1309]
	tl = [1675, 35]
	bl = [119, 5964]

	# brt = np.dot(M, br); br = [brt[0]/brt[2], brt[1]/brt[2]]
	# trt = np.dot(M, tr); p1 = [trt[0]/trt[2], trt[1]/trt[2]]
	# tlt = np.dot(M, tl); p1 = [tlt[0]/tlt[2], tlt[1]/tlt[2]]
	# blt = np.dot(M, bl); p1 = [blt[0]/blt[2], blt[1]/blt[2]]



	# l1 = (tl, bl)
	l1_m =  (bl[1] - tl[1])/(bl[0] - tl[0])
	l1_b = bl[1] - (l1_m*bl[0])

	# l2 = (tr, br)
	l2_m =  (br[1] - tr[1])/(br[0] - tr[0])
	l2_b = br[1] - (l1_m*br[0])

	# l3 = (tl, tr)
	l3_m =  (tr[1] - tl[1])/(tr[0] - tl[0])
	l3_b = tr[1] - (l3_m*tr[0])

	# l4 = (bl, br)
	l4_m =  (br[1] - bl[1])/(br[0] - bl[0])
	l4_b = br[1] - (l4_m*br[0])

	l1_del = abs(bl[1] - tl[1])/grid_size
	l1_x = [tl[0]]
	l1_y = [tl[1]] 

	for val in range(1, grid_size):
		y = int(tl[1] + val*l1_del)
		l1_y.append(y)
		x = (y - l1_b)/l1_m
		l1_x.append(int(x))

	l2_del = abs(br[1] - tr[1])/grid_size
	l2_x = [tr[0]]
	l2_y = [tr[1]] 
	
	for val in range(1, grid_size):
		y = int(tr[1] + val*l2_del)
		l2_y.append(y)
		x = (y - l2_b)/l2_m
		l2_x.append(int(x))

	l3_del = abs(tr[0] - tl[0])/grid_size
	l3_x = [tl[0]]
	l3_y = [tl[1]] 
	
	for val in range(1, grid_size):
		x = int(tl[0] + val*l3_del)
		l3_x.append(x)
		y = (l3_m*x) + l3_b
		l3_y.append(int(y))

	l4_del = abs(br[0] - bl[0])/grid_size
	l4_x = [bl[0]]
	l4_y = [bl[1]] 

	for val in range(1, grid_size):
		x = int(bl[0] + val*l4_del)
		l4_x.append(x)
		y = (l4_m*x) + l4_b
		l4_y.append(int(y))

	
	for i in range(grid_size):
		line1 = Line2D([l1_x[i], l2_x[i]],[l1_y[i], l2_y[i]], linestyle="dashed", linewidth=0.5,color="black")	
		line2 = Line2D([l3_x[i], l4_x[i]],[l3_y[i], l4_y[i]], linestyle="dashed",linewidth=0.5, color="black")
		ax1.add_line(line1)
		ax1.add_line(line2)


	for i in range(len(l1_x)):

		p1 = [l1_x[i], l1_y[i], 1]
		p2 = [l2_x[i], l2_y[i], 1]
		p3 = [l3_x[i], l3_y[i], 1]
		p4 = [l4_x[i], l4_y[i], 1]
		p1 = np.dot(M, p1); p1 = [p1[0]/p1[2], p1[1]/p1[2]]
		p2 = np.dot(M, p2); p2 = [p2[0]/p2[2], p2[1]/p2[2]]
		p3 = np.dot(M, p3); p3 = [p3[0]/p3[2], p3[1]/p3[2]]
		p4 = np.dot(M, p4); p4 = [p4[0]/p4[2], p4[1]/p4[2]]

		line1 = Line2D([p1[0], p2[0]], [p1[1], p2[1]], linestyle="dashed",linewidth=0.5, color="black")
		line2 = Line2D([p3[0], p4[0]], [p3[1], p4[1]], linestyle="dashed",linewidth=0.5, color="black")
		ax2.add_line(line1)
		ax2.add_line(line2)



def plot_lines(src, dst, M):

	f = plt.figure('rec_img')
	g = plt.figure('rec_raw_img')
	ax1 = f.add_subplot(111)
	ax2 = g.add_subplot(111)
	
	r, c, p = src.shape 
	grid_size = 10
	
	generate_lat_long(r, c, ax1, ax2, M, grid_size)
	
	ax1.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
	ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
	plt.show()


# src=mpimg.imread('fcc.png')
# dst=mpimg.imread('fcc_rot.png')
# M = np.array([[  0.965925826,   0.258819045,  -806.529614],[ -0.258819045,   0.965925826,   1164.79937],[0, 0, 1]]) #curr
# plot_lines(src, dst, M)
