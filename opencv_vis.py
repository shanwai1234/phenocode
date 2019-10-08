import numpy as np
import cv2 
from matplotlib import pyplot as plt
import sys

#back_im = cv2.imread("Day_001.png")
#for_im = cv2.imread('Day_025.png')

#abc = cv2.subtract(back_im,for_im)
abc
imgray = cv2.cvtColor(abc,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,64,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
areas = []
contours2 = []
for c in contours:
	areas.append(cv2.contourArea(c))
	if areas[-1] > 1000: contours2.append(c)

people = np.array(contours)
ages = np.array(areas)
inds = ages.argsort()
sortedcontours = people[inds]


cnt = sortedcontours[-1]

hull = cv2.convexHull(cnt)

x,y,w,h = cv2.boundingRect(cnt)
(x2,y2),radius = cv2.minEnclosingCircle(cnt)

center = (int(x2),int(y2))
radius = int(radius)

print "Convex Hull Area: {0} pixels".format(cv2.contourArea(hull))
print "Plant Height: {0} pixels".format(h)
print x,y,w,h
cv2.drawContours(abc, contours2, -1, (0,255,0), 5)
cv2.rectangle(abc,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),2)
cv2.circle(abc,center,radius,(255,0,0),2)

cv2.imwrite("see_this2.jpg", abc)
cv2.imwrite("gray.jpg",thresh)
