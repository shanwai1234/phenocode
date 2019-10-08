import numpy as np
import cv2 
from matplotlib import pyplot as plt
import os
import sys
import numpy as NP
from scipy import linalg as LA
from matplotlib  import cm
hyp_name = "HYP SV 90"

def PCA2(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs

def plot_pca(data):
    from matplotlib import pyplot as MPL
    clr1 =  '#2026B2'
    fig = MPL.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig,a = PCA2(data)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
    MPL.show()
    return data_resc

# create a funtion to binary segmented photos
def binary(pic):
	final = []
	myl = np.shape(pic)[0]
	myw = np.shape(pic)[1]
	for i in pic:
		n = []
		for j in i:
			# only for each pixel that greater than 0.25 were converted to whole white pixel, while pixel value less than that was converted to black
			if j > 0.25:
				t = 255
			else:
				t = 0
			n.append(t)
		final.append(n)
	final = np.array(final)	
	return final

# create a funtion to remove stem
def rmstem(pic1,pic2):
	mypic = []
	myl = np.shape(pic1)[0]
	myw = np.shape(pic1)[1]
	for i in range(myl):
		n = []
		for j in range(myw):
			if pic1[i,j] == 0 or pic2[i,j] == 0:
				n.append(0)
			else:
				ratio = pic1[i,j]/pic2[i,j]
				if ratio > 1.2:
					n.append(255)
				else:
					n.append(0)
		mypic.append(n)
	mypic = np.array(mypic)
	return mypic

# create a function to merge first binaried image and stem removed image together
def merge(pic1,pic2):
	final = []
	myl = np.shape(pic1)[0]
	myw = np.shape(pic1)[1]
	for i in range(myl):
		n = []
		for j in range(myw):
			diff = pic1[i,j]-pic2[i,j]
			if diff <= 0:
				n.append(0)
			else:
				n.append(255)
		final.append(n)
	final = np.array(final)
	return final

def early(pic):
	final = []
	myl = np.shape(pic)[0]
	myw = np.shape(pic)[1]
	for i in range(myl):
		n = []
		for j in range(myw):
			if i < 40 or i > 460 or j < 40 or j > 250:
				n.append(0)
			else:
				n.append(pic[i,j])
		final.append(n)
	final = np.array(final)
	return final

def PCA(pic1, pic2,n):
	final = {}
	myl = np.shape(pic1)[0]
	myw = np.shape(pic1)[1]
	for i in range(myl):
		for j in range(myw):
			if pic1[i,j] == 255:
                                myname = "{0}-{1}-{2}".format(i,j,n)
                                final[myname] = pic2[i,j]
#				final.append(pic2[i,j])
	return final

# sh is the reference file showing which file corresponds to which wavelength
sh = open('wavelength_foldid.txt','r')
sh.readline()
kdict = {}
# build a library to include file~wavelength information
for line in sh:
	new = line.strip().split('\t')
	kdict[new[4]] = new[0]

first9 = set([])
for i in range(13,32):
	first9.add('2015'+'-'+'10'+'-'+str(i))
for i in range(1,10):
	first9.add('2015'+'-'+'11'+'-'+'0'+str(i))
first9.add('2015-11-10')
first3 = set([])
for i in range(10,13):
	first3.add('2015'+'-'+'10'+'-'+str(i))

ll = []
whole = os.listdir('test_PCA')
mdict = {}
tlist = []
for j1 in whole:
#	if j1 != "10-5-15 maize_108-166-21_2015-11-08_10-15-34_3607700":continue
#	if j1 != "10-5-15 maize_108-152-29_2015-11-08_09-41-54_3606300":continue
#	if j1 != "10-5-15 maize_108-170-18_2015-11-08_10-22-39_3608100":continue
#	mdict = {}
	myid = j1.split('_')[1]
	plant = myid.split('-')[1]
	tlist.append(plant)
#	if not myid.startswith('108'):continue
	date = j1.split('_')[2]
#	if myid == '108-087-5' and date == '2015-10-14':continue
#	if date in first3:continue
	full = myid + '_' + date
	subset = 'Maize_diversity/{0}/{1}/'.format(j1,hyp_name)
	# in every folder, the images of 35_0_0.png and 45_0_0.png should be used firstly in order to subtract the plant area
	if True:
		m705 = cv2.imread("Maize_diversity/{0}/{1}/35_0_0.png".format(j1,hyp_name))	
		m750 = cv2.imread("Maize_diversity/{0}/{1}/45_0_0.png".format(j1,hyp_name))
		# these files are used to remove stem area in one plant
		m1056 = cv2.imread("Maize_diversity/{0}/{1}/108_0_0.png".format(j1,hyp_name))
		m1151 = cv2.imread("Maize_diversity/{0}/{1}/128_0_0.png".format(j1,hyp_name))
		# converting plant images from RGB to GRAY channel
		tm705 = cv2.cvtColor(m705,cv2.COLOR_BGR2GRAY)
		tm750 = cv2.cvtColor(m750,cv2.COLOR_BGR2GRAY)
		
		tm1056 = cv2.cvtColor(m1056,cv2.COLOR_BGR2GRAY)
		tm1151 = cv2.cvtColor(m1151,cv2.COLOR_BGR2GRAY)
	
		tm1056 = tm1056.astype(np.float)
		tm1151 = tm1151.astype(np.float)
		rmg = rmstem(tm1056,tm1151)
		# This is going to print out the image and help to check if the binarization process works well, but can skip that in bundle of processing.
		if date in first9:
			rmg = early(rmg)
#		cv2.imwrite('binary.jpg',mmg)
		cv2.imwrite('stem.jpg',rmg)
		cv2.imwrite('{0}.jpg'.format(full),rmg)
		for i in os.listdir(subset):
			# first two images are not useful and just skip them
			if i == '0_0_0.png':continue
			if i == '1_0_0.png':continue
			# info.txt is not an image file
			if i == 'info.txt':continue
			name = i.replace('_0_0.png','')
			try:
				t = cv2.imread("Maize_diversity/{0}/{2}/{1}".format(j1,i,hyp_name))
				t = t.astype(np.float)
				t1 = t[:,:,0]
				# multiply each files in the folder with the binarized image. For each pixel, dividing 255 to make each pixel in 0~1 
				# t2 = np.multiply(t1,rmg)
#				total = []
				total = PCA(rmg,t1,plant)
				if name not in mdict:
					mdict[name] = {}
				mdict[name].update(total)
			#	print mdict
			except:
				print j1+':'+i
                wavelengths = list(mdict)
                pixels = list(mdict[wavelengths[0]])
		out = open('test_purpleness/{0}'.format(full),'w')
		for i in range(2,245):
			i = str(i)
			if i not in mdict:continue
			out.write(kdict[i]+','+','.join(map(str,mdict[i]))+'\n')
	else:
		print j1

for p in pixels:
	ll.append([])
	for w in wavelengths:				
		ll[-1].append(mdict[w][p])
ll_array = NP.array(ll)
data_resc = plot_pca(ll_array)

for x in range(2):
	mytitle = "PC {0}".format(x+1)
	fig = plt.figure()
	ax = fig.add_subplot('111')
	ax.set_title(mytitle)
	myxvals = {}
	myyvals = {}
	mycvals = {}
	for name,val in zip(pixels,data_resc[:,x]):
		l = map(int,name.split('-')[:2])
		myid = name.split('-')[2]
		if myid not in myxvals:
			myxvals[myid] = []
			myyvals[myid] = []
			mycvals[myid] = []
		myyvals[myid].append(l[0]*(-1))
		myxvals[myid].append(l[1])
		mycvals[myid].append(val)
	
	n = 0
	myxtick = []
	myxname = []
	for i in myxvals:
		myxname.append(i)
		myxtick.append(NP.median(myxvals[i])+n*50)
		ax.scatter([x+n*50 for x in myxvals[i]],myyvals[i],c=mycvals[i],marker='o',cmap = cm.seismic)
		n += 1
	plt.xlim([40,240+n*50])
	ax.set_xticks(myxtick)
	ax.set_xticklabels(myxname)
	ax.set_yticklabels([])
	plt.show()
