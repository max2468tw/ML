import numpy as np
import sys

data = np.loadtxt('train.csv', delimiter = ',', dtype={'names':('index','0','1','2','3','4','5','6','7','8','9','10'),'formats': ('i4','U15','f8','f8','f8','f8','f8','f8','f8','f8','f8','U5')})

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
 
root = Tree()
root.data = "root"
root.left = Tree()
root.left.data = "left"
root.right = Tree()
root.right.data = "right"
 
print(root.left.data)
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
	
def get_next_attribute(attribute):
	if attribute == '9':
		return '1'
	else:
		return str(int(attribute) + 1)
		
def choose_middle(data, attribute):
	data = np.sort(data, order = attribute) 
	if data.size % 2 == 1:
		mid = (data.size + 1)/2
	else:
		mid = (data.size)/2
	mid = int(mid)
	return data[mid - 1]

def create_kdtree(data, attribute):
	if data.size == 0:
		return None
	elif data.size == 1:
		return data[0]
	else:		
		best = choose_middle(data,attribute)
		root = Tree()
		root.data = best
		data0 = data1 = np.array([],dtype = data.dtype)
		for i in range(data.size):
			if data[i]['index'] != best['index']:
				if data[i][attribute] < best[attribute]:
					data0 = np.append(data0,data[i])
				else:
					data1 = np.append(data1,data[i])		
		get_next_attribute(attribute)
		subtree = create_kdtree(data0, attribute)
		root.left = subtree
		subtree = create_kdtree(data1, attribute)
		root.right = subtree
	return root

def descendTree(x, q, attribute):
	if x.left == None && x.right == None:
		return [x,attribute]
	elif q[attribute] >= x[attribute]:
		if x.right != None:
			get_next_attribute(attribute)
			return descendTree(x.right, q, attribute)
		else:
			return [x,attribute]
	else:
		if x.left != None:
			get_next_attribute(attribute)
			return descendTree(x.left, q, attribute)
		else:
			return [x,attribute]

def find_the_nearest(q, r):
	t = None
	d_t = 10000000000000
	x = descendTree(r,q,'1')
	while x != None:
		if d(q,x) < d_t:
			t = x
			d_t = d(q,x)
		if boundaryDist(q,x) < d_t:
			x = descendTree(x,q)
		else:
			x = parent(x)
	return t

