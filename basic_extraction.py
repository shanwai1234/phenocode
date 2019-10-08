import numpy as np
import cv2 
from matplotlib import pyplot as plt
import sys


def binary(pic,upper,bottom,left,right):
	mypic = []
	myl = np.shape(pic)[0]
	myw = np.shape(pic)[1]
	x1 = left
	x2 = right
	y1 = upper
	y2 = bottom
	for iind,i in enumerate(pic):
		if iind < y1 or iind > y2:
			n = [0]*myw
		else:
			n = []
			for jind,j in enumerate(i):
				if j > 1.15:
					if jind < x1 or jind > x2:
						t = 0
					else:
						t = 255
				else:
					t = 0 
				n.append(t)
		mypic.append(n)
	mypic = np.array(mypic)
	return mypic

def call_numeric(thresh):
	hh = 0
	ww = 0
	aa = 0
	contours,hierarchy = cv2.findContours(thresh, 1, 2)
	areas = []
	for c in contours:
		areas.append(cv2.contourArea(c))
	people = np.array(contours)
	ages = np.array(areas)
	inds = ages.argsort()
	sortedcontours = people[inds]
	cnt = sortedcontours[-1]
	hull = cv2.convexHull(cnt)
	x,y,w,h = cv2.boundingRect(cnt)
	hh = h
	ww = w
	aa = cv2.contourArea(hull)
	return hh,ww,aa

a = sys.argv[1]
myfold = os.listdir(a)
close = set([])
far = set([])
for i in range(10,32):
	close.add('2015-10-{0}'.format(i))
for i in range(1,5):
	close.add('2015-11-{0}'.format(i.zfill(2)))
close.remove('2015-10-20')
for i in range(5,11):
	far.add('2015-11-{0}'.format(i.zfill(2)))
far.add('2015-10-20')
out = open('exp2_green_extraction_results.csv','r')
out.write('PlantID'+'\t'+'View'+'\t'+'Plant Height'+'\t'+'Plant Width'+'\t'+'Convex Hull Area'+'\n')
for f in myfold:
	cc = f.split('_')
	plant = cc[1]
	date = cc[2]
	nlist = [plant]
	try:
		abc0 = cv2.imread('{0}/{1}/VIS SV 0/0_0_0.png'.format(a,f))
		abc0 = abc0.astype(np.float)
		imgreen0 = (2*abc0[:,:,1])/(abc0[:,:,0]+abc0[:,:,2])
		if date in close:
			thresh0 = binary(imgreen0,50,1950,335,2280)
		elif date in far:
			thresh0 = binary(imgreen0,50,1450,815,1170)
		cv2.imwrite('test.jpg',thresh0)
		thresh0 = cv2.imread("test.jpg",cv2.CV_LOAD_IMAGE_GRAYSCALE)
		h0,w0,area0 = call_numeric(thresh0)
		nlist.append('0 View')
		nlist.append(h0)
		nlist.append(w0)
		nlist.append(area0)
	except:
		print f
	try:
		abc90 = cv2.imread('{0}/{1}/VIS SV 90/0_0_0.png'.format(a,f))
		abc90 = abc90.astype(np.float)
		imgreen90 = (2*abc90[:,:,1])/(abc90[:,:,0]+abc90[:,:,2])
		if date in close:
			thresh90 = binary(imgreen90,50,1950,335,2280)
		elif date in far:
			thresh90 = binary(imgreen90,50,1450,815,1170)
		cv2.imwrite('test.jpg',thresh90)
		thresh90 = cv2.imread("test.jpg",cv2.CV_LOAD_IMAGE_GRAYSCALE)
		h90,w90,area90 = call_numeric(thresh90)
		nlist.append('90 View')
		nlist.append(h90)
		nlist.append(w90)
		nlist.append(area90)
	except:
		print f
	out.write('\t'.join(nlist)+'\n')
out.close()
