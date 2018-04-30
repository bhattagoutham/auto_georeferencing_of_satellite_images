import matplotlib.pyplot as plt
from matplotlib.lines import Line2D      
import matplotlib.image as mpimg
import numpy as np
from numpy.linalg import inv
import cv2
import sys

def gen_lat_long(r, c, grid_size):
	# get list of points for all the lat_y_pts
	lat_y = []
	for val in range(1,grid_size):
		temp = int(val*(r/grid_size))
		lat_y.append(temp)

	# get list of points for all the long_y_pts
	long_x = []
	for val in range(1,grid_size):
		temp = int(val*(c/grid_size))
		long_x.append(temp)

	return lat_y, long_x

def plot_lat_long(r, c, lat_y, long_x, ax):

	lat_x = [0, c]
	for y in lat_y:
		line = Line2D(lat_x, [y, y])	
		ax.add_line(line)

	long_y = [0, r]
	for x in long_x:
		line = Line2D([x, x], long_y)	
		ax.add_line(line)

	return ax

def transform_lat_long(r, c, lat_y, long_x, M, ax):

	for y in lat_y:
		p1 = [0, y, 1]; p2 = [c, y, 1]
		p1 = np.dot(M, p1); p1 = [p1[0]/p1[2], p1[1]/p1[2]]
		p2 = np.dot(M, p2); p2 = [p2[0]/p2[2], p2[1]/p2[2]]
		line = Line2D([p1[0], p2[0]], [p1[1], p2[1]])
		ax.add_line(line)
	
	for x in long_x:
		p1 = [x, 0, 1]; p2 = [x, r, 1]
		p1 = np.dot(M, p1); p1 = [p1[0]/p1[2], p1[1]/p1[2]]
		p2 = np.dot(M, p2); p2 = [p2[0]/p2[2], p2[1]/p2[2]]
		line = Line2D([p1[0], p2[0]], [p1[1], p2[1]])
		ax.add_line(line)

	return ax


def plot_lines(src,dst,M ):

	f = plt.figure('rec_img')
	ax1 = f.add_subplot(111)
	r, c, p = src.shape 
	grid_size = 10
	lat_y, long_x = gen_lat_long(r, c, grid_size)
	ax1 = plot_lat_long(r, c, lat_y, long_x, ax1)
	ax1.imshow(src, cmap='gray')

	# im_out = cv2.warpPerspective(src, M,(c, r))
	g = plt.figure('rec_raw_img')
	ax2 = g.add_subplot(111)
	ax2 = transform_lat_long(r, c, lat_y, long_x, M, ax2)
	ax2.imshow(dst, cmap='gray')
	plt.show(block=True)


