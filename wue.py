import sys

a = sys.argv[1]
b = sys.argv[2]

fh = open(a,'r')
out = open(b,'w')
fh.readline()

water = {}
diff = {}
myy = {}
only = set([])
out.write('plantID,GeneticID,Date,WUE'+'\n')
for line in fh:
	new = line.strip().split(',')
	if new[5].startswith('N'):
		new[5] = 0
	if new[6].startswith('N'):
		new[6] = 0
	if new[1] not in only:
		only.add(new[1])
		water[new[1]] = []
		myy[new[1]] = []
		diff[new[1]] = []
	water[new[1]].append(float(new[6])-float(new[5]))
	myy[new[1]].append(float(new[-2]))
	if len(myy[new[1]]) > 1 and len(water[new[1]]) > 1:
		diff[new[1]].append(new)
		if water[new[1]][-2] == 0:
			diff[new[1]].append(0)
		else:
		# here using (Dry Weight - last Dry Weight)/(amount of water last date)
			ef = (float(new[-2])-myy[new[1]][-2])/(water[new[1]][-2])
			if abs(ef) > 0.02:
				ef = 0
				diff[new[1]].append(ef)
			else:
				diff[new[1]].append(ef)

for i in diff:
	n = 2
	for k in range(0,len(diff[i]),2):
		out.write(str(diff[i][k][1])+','+str(diff[i][k][2])+','+str(n)+','+str(diff[i][k][3])+','+str(diff[i][k+1])+'\n')
		n += 1	

fh.close()
out.close()
