import numpy as np
import sys
from numpy import linalg as LA
from math import sqrt
from copy import deepcopy

class Tree(object):
	def __init__(self):
		self.parent = None
		self.left = None
		self.right = None
		self.data = None

def normalize(data):
	normalized_data = data
	names = np.asarray(data.dtype.names)
	for i in range(names.size):
		if i != 0 and i != 1 and i != names.size-1:
			max = 0
			min = sys.float_info.max
			for j in range(data.size):
				if data[names[i]][j] > max:
					max = data[names[i]][j]
				if data[names[i]][j] < min:
					min = data[names[i]][j]
			for j in range(data.size):
				normalized_data[names[i]][j] = (data[names[i]][j] - min)/(max-min)
	return normalized_data

def get_next_attribute(attribute):
	if attribute == '9':
		return '1'
	else:
		return str(int(attribute) + 1)

def get_next_attribute_PCA(attribute):
	if attribute == '7':
		return '1'
	else:
		return str(int(attribute) + 1)
		
def choose_middle(data, attribute):
	data = np.sort(data, order = attribute) 
	if data.size % 2 == 1:
		mid = (data.size + 1)/2
	else:
		mid = (data.size)/2
	mid = int(mid - 1)
	return data[mid]

def create_kdtree(data, attribute):
	if data.size == 0:
		return None
	elif data.size == 1:
		root = Tree()
		root.data = [data[0], attribute]
		return root
	else:		
		best = choose_middle(data,attribute)
		root = Tree()
		root.data = [best,attribute]
		data0 = data1 = np.array([],dtype = data.dtype)
		for i in range(data.size):
			if data[i] != best:
				if data[i][attribute] < best[attribute]:
					data0 = np.append(data0,data[i])
				else:
					data1 = np.append(data1,data[i])		
		attribute = get_next_attribute(attribute)
		subtree = create_kdtree(data0, attribute)
		root.left = subtree
		if subtree != None:
			subtree.parent = root
		subtree = create_kdtree(data1, attribute)
		root.right = subtree
		if subtree != None:
			subtree.parent = root
	return root

def create_kdtree_PCA(data, attribute):
	if data.size == 0:
		return None
	elif data.size == 1:
		root = Tree()
		root.data = [data[0], attribute]
		return root
	else:		
		best = choose_middle(data,attribute)
		root = Tree()
		root.data = [best,attribute]
		data0 = data1 = np.array([],dtype = data.dtype)
		for i in range(data.size):
			if data[i] != best:
				if data[i][attribute] < best[attribute]:
					data0 = np.append(data0,data[i])
				else:
					data1 = np.append(data1,data[i])		
		attribute = get_next_attribute_PCA(attribute)
		subtree = create_kdtree_PCA(data0, attribute)
		root.left = subtree
		if subtree != None:
			subtree.parent = root
		subtree = create_kdtree_PCA(data1, attribute)
		root.right = subtree
		if subtree != None:
			subtree.parent = root
	return root

def d(q, x):
	sum = 0
	names = np.asarray(x.data[0].dtype.names)
	for i in range(names.size):
		if i != 0 and i != 1 and i != names.size-1:
			sum = sum + (x.data[0][names[i]] - q[names[i]]) * (x.data[0][names[i]] - q[names[i]])
	return sqrt(sum)

def descendTree(x, q):
	if x == None:
		return None
	if x.left == None and x.right == None:
		return x
	elif q[x.data[1]] >= x.data[0][x.data[1]]:
		if x.right != None:
			return descendTree(x.right, q)
		else:
			return descendTree(x.left, q)
	else:
		if x.left != None:
			return descendTree(x.left, q)
		else:
			return descendTree(x.right, q)

def parent(x):
	parent = x.parent
	if parent == None:
		return None
	if parent.left != None and x == parent.left:
		parent.left = None
	elif parent.right != None and x == parent.right:
		parent.right = None
	return parent

def boundaryDist(q, x):
	return q[x.data[1]] - x.data[0][x.data[1]]

def find_the_nearest(q, r):
	t = None
	d_t = 2147483647
	x = descendTree(r,q)
	while x != None:
		if d(q,x) < d_t:
			t = x
			d_t = d(q,x)
		if x.left != None and x.right != None and boundaryDist(q,x) < d_t:
			x = descendTree(x,q)
		else:
			x = parent(x)
	return t

def check_result(x):
	if x == "cp":
		return 0
	elif x == "im":
		return 1
	elif x == "pp":
		return 2
	elif x == "imU":
		return 3
	elif x == "om":
		return 4
	elif x == "omL":
		return 5
	elif x == "imL":
		return 6
	elif x == "imS":
		return 7

def KNN(trainning_data, testing_data, k, list):
	tmp_data = trainning_data
	trace = [0,0,0,0,0,0,0,0]
	for i in range(k):
		tree = create_kdtree(tmp_data, '1')
		x = find_the_nearest(testing_data,tree)	
		list = np.append(list, x.data[0]['index'])
		trace[check_result(x.data[0]['10'])] = trace[check_result(x.data[0]['10'])] + 1
		for j in range(tmp_data.size):
			if tmp_data[j] == x.data[0]:
				tmp_data = np.delete(tmp_data,j)
				break
	max = 0
	max_id = 0
	for i in range(len(trace)):
		if trace[i] > max:
			max = trace[i]
			max_id = i
	return [max_id, list]	 


def KNN_PCA(trainning_data, testing_data, k, list):
	tmp_data = trainning_data
	trace = [0,0,0,0,0,0,0,0]
	for i in range(k):
		tree = create_kdtree_PCA(tmp_data, '1')
		x = find_the_nearest(testing_data,tree)	
		list = np.append(list, x.data[0]['index'])
		trace[check_result(x.data[0]['10'])] = trace[check_result(x.data[0]['10'])] + 1
		for j in range(tmp_data.size):
			if tmp_data[j] == x.data[0]:
				tmp_data = np.delete(tmp_data,j)
				break
	max = 0
	max_id = 0
	for i in range(len(trace)):
		if trace[i] > max:
			max = trace[i]
			max_id = i
	return [max_id, list]

def main(k):
	with open ('train.csv', 'r') as f:
		a = f.readlines()
		del a[0]
	with open('tr.csv', 'w') as f:
		for i in range(len(a)):		
			f.write(a[i])
	with open ('test.csv', 'r') as f:
		a = f.readlines()
		del a[0]
	with open('ts.csv', 'w') as f:
		for i in range(len(a)):		
			f.write(a[i])
	trainning_data = np.loadtxt('tr.csv', delimiter = ',', dtype={'names':('index','0','1','2','3','4','5','6','7','8','9','10'),'formats': ('i4','U15','f8','f8','f8','f8','f8','f8','f8','f8','f8','U5')})
	testing_data = np.loadtxt('ts.csv', delimiter = ',', dtype={'names':('index','0','1','2','3','4','5','6','7','8','9','10'),'formats': ('i4','U15','f8','f8','f8','f8','f8','f8','f8','f8','f8','U5')})
	t = f = 0
	list = list1 = list2 = list3 = np.array([])
	print('KNN accuracy:', end = " ")
	for i in range(testing_data.size):
		if i == 0:
			x = KNN(trainning_data,testing_data[i], k, list1)
			if check_result(testing_data[i]['10']) == x[0]:
				list1 = x[1]
				t = t + 1
			else:
				list1 = x[1]
				f = f + 1
		elif i == 1:
			x = KNN(trainning_data,testing_data[i],k, list2)
			if check_result(testing_data[i]['10']) == x[0]:
				list2 = x[1]
				t = t + 1
			else:
				list2 = x[1]
				f = f + 1
		elif i == 2:
			x = KNN(trainning_data,testing_data[i],k, list3)
			if check_result(testing_data[i]['10']) == x[0]:
				list3 = x[1]
				t = t + 1
			else:
				list3 = x[1]
				f = f + 1
		else:		
			if check_result(testing_data[i]['10']) == KNN(trainning_data,testing_data[i],k, list)[0]:
				t = t + 1
			else:
				f = f + 1
	print(t/(t+f))
	print(list1)
	print(list2)
	print(list3)
	print()
	return

def PCA():
	D = np.loadtxt('tr.csv', delimiter = ',', dtype={'names':('index','0','1','2','3','4','5','6','7','8','9','10'),'formats': ('i4','U15','f8','f8','f8','f8','f8','f8','f8','f8','f8','U5')})
	T = np.loadtxt('ts.csv', delimiter = ',', dtype={'names':('index','0','1','2','3','4','5','6','7','8','9','10'),'formats': ('i4','U15','f8','f8','f8','f8','f8','f8','f8','f8','f8','U5')})	
	At = np.array([])
	At = np.append(At,D['1'])
	At = np.append(At,D['2'])
	At = np.append(At,D['3'])
	At = np.append(At,D['4'])
	At = np.append(At,D['5'])
	At = np.append(At,D['6'])
	At = np.append(At,D['7'])
	At = np.append(At,D['8'])
	At = np.append(At,D['9'])
	At = np.reshape(At,(9,300))
	x_avg = np.array([])
	for i in range(9):
		avg = np.average(At[i])
		At[i] = At[i] - avg
		x_avg = np.append(x_avg, avg)
	A = np.transpose(At)
	dot = np.dot(At,A)
	w, v = LA.eig(dot)
	eig = np.sum(w)
	ts = 0.98
	for i in range(w.size - 1):
		for j in range(w.size - i - 1):
			if w[j] < w[j+1]:
				swap = w[j]
				w[j] = w[j+1]
				w[j+1] = swap
				tmp = deepcopy(v[j])
				v[j] = v[j+1]
				v[j+1] = tmp
	sum = 0
	cnt = 0
	Q = np.array([])
	for i in range(w.size):
		if sum/eig > ts:
			break
		else:
			Q = np.append(Q, v[i])
			sum = sum + w[i]
			cnt = cnt + 1
	Q = np.reshape(Q,(cnt,9))
	Q = np.transpose(Q)
	PCA_data = np.dot(A,Q)
	print('K = 5, KNN_PCA accuracy:', end = " ")
	test = np.array([])
	test = np.append(test,T['1'])
	test = np.append(test,T['2'])
	test = np.append(test,T['3'])
	test = np.append(test,T['4'])
	test = np.append(test,T['5'])
	test = np.append(test,T['6'])
	test = np.append(test,T['7'])
	test = np.append(test,T['8'])
	test = np.append(test,T['9'])
	test = np.reshape(test,(9,36))
	test = np.transpose(test)
	x_avg = np.reshape(x_avg,(1,9))
	test = np.subtract(test,x_avg)
	PCA_test = np.dot(test,Q)
	list = np.array([])
	PCA_T = np.array([],dtype={'names':('index','1','2','3','4','5','6','7','10'),'formats': ('i4','f8','f8','f8','f8','f8','f8','f8', 'U5')})	
	for i in range(36):
		PCA_T = np.append(PCA_T, np.array([(i,PCA_test[i][0],PCA_test[i][1],PCA_test[i][2],PCA_test[i][3],PCA_test[i][4],PCA_test[i][5],PCA_test[i][6],T[i]['10'])], dtype=PCA_T.dtype))
	PCA_D = np.array([], dtype=PCA_T.dtype)
	for i in range(300):
		PCA_D = np.append(PCA_D, np.array([(i,PCA_data[i][0],PCA_data[i][1],PCA_data[i][2],PCA_data[i][3],PCA_data[i][4],PCA_data[i][5],PCA_data[i][6],D[i]['10'])], dtype=PCA_T.dtype))
	t = f = 1
	for i in range(36):			
		if check_result(PCA_T[i]['10']) == KNN_PCA(PCA_D, PCA_T[i], 5, list)[0]:
			t = t + 1
		else:
			f = f + 1
	print(t/(t+f))
	return

main(1)
main(5)
main(10)
main(100)
PCA()
