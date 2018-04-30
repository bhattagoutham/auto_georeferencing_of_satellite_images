import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from threading import Thread
import math as mt
import sys
from lines import plot_lines


h = []
def get_input():
	global h
	src_pts = []
	dst_pts = []
	with open('coords.txt', 'r') as f:
		read_data = f.read()
		coords = read_data.split('\n')
	f.close()

	for temp in coords:
		if temp:
			src, dst = temp.split(';')
			s_x, s_y = src.split(',')
			d_x, d_y = dst.split(',')
			src_pts.append([float(s_x.strip()), float(s_y.strip())])
			dst_pts.append([float(d_x.strip()), float(d_y.strip())])

	src_pts = 	np.array(src_pts)
	dst_pts = 	np.array(dst_pts)
	
	h, status = cv2.findHomography(src_pts, dst_pts)


	print('Enter additional co-ordinates...')
	
	while(1):
		
		print('Enter rec_img_co-ordinate:(q for quit) ', end=' ')
		src = input()
		if src == 'q':
			sys.exit(0)

		
		s_x, s_y = src.split(',')
		src_pt = np.transpose(np.array([float(s_x.strip()), float(s_y.strip()), 1]))
		
		res = np.dot(h, src_pt)
		res = np.array([res[0]/res[2], res[1]/res[2]])
		print("Expected raw_img_pt :", res)

		print('Enter raw_img_co-ordinate:', end=' ')
		dst = input()
		if src == 'q':
			sys.exit(0)
		
		d_x, d_y = dst.split(',')
		dst_pt = np.transpose(np.array([float(d_x.strip()), float(d_y.strip())]))
		
		rsme = mt.sqrt(sum((res - dst_pt)**2))
		print('RSME:(sigma)', rsme)

    


def affine_transform():
	
	t = Thread(target=get_input)
	t.start()
	
	src=mpimg.imread('rec_img.png')
	dst=mpimg.imread('raw_img.png')
	
	f = plt.figure('rec_img')
	plt.imshow(src, cmap='gray')
	f.show()

	g = plt.figure('raw_img')
	plt.imshow(dst, cmap='gray')
	g.show()

	plt.show(block=True)

	t.join()
	
	plot_lines(src, dst, h)



affine_transform()


