import numpy as np
import sys

data = np.loadtxt('train.csv', delimiter = ',', dtype={'names':('index','0','1','2','3','4','5','6','7','8','9','10'),'formats': ('i4','U15','f8','f8','f8','f8','f8','f8','f8','f8','f8','U5')})

def normalize(data):
	normalized_data = data
	names = np.asarray(data.dtype.names)
	for i in range(names.size-1):
		if i != 0 and i != 1 and i!=names.size-1:
			max = 0
			min = sys.float_info.max
			for j in range(data.size-1):
				if data[names[i]][j] > max:
					max = data[names[i]][j]
				if data[names[i]][j] < min:
					min = data[names[i]][j]
			for j in range(data.size-1):
				normalized_data[names[i]][j] = (data[names[i]][j] - min)/(max-min)
	return normalized_data

def choose_middle(data, attributes):
	data = np.sort(data, order = attributes) 
	if data.size % 2 == 1:
		mid = (data.size + 1)/2
	else:
		mid = (data.size)/2
	mid = int(mid)
	return data[mid - 1]

def create_kdtree(data, attributes):
	target = data['index']
	if data.size == 0:
		return -1
	elif data.size == 1:
		print(target[0])
		return target[0]
	else:		
		best = choose_middle(data,attributes)
		tree = {best['index']:{}}
		data0 = data1 = np.array([],dtype = data.dtype)
		for i in range(data.size):
			if data[i]['index'] != best['index']:
				if data[i][attributes] < best[attributes]:
					data0 = np.append(data0,data[i])
				else:
					data1 = np.append(data1,data[i])		
		if attributes == '9':
			attributes = '1'
		else:
			attributes = str(int(attributes) + 1)
		subtree = create_kdtree(data0, attributes)
		tree[best['index']][0] = subtree
		subtree = create_kdtree(data1, attributes)
		tree[best['index']][1] = subtree
	return tree

def descendTree(x, q, data):
	attributes = '1'	
	while(1):
		for attr in x.keys():
			if q[i] < ts:
				x = x[attr][0]
			else:	
				x = x[attr][1]
