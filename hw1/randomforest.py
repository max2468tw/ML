import numpy as np
from math import log2

#loadtxt from the data file and save as a numpy array
data = np.loadtxt('data', delimiter = ',', dtype={'names':('sepal length','sepal width','petal length','petal width','class'),'formats': ('f8','f8','f8','f8','U20')})

#shuffle the data
np.random.shuffle(data)

def get_attributes(data):
	#sorting the data with the different attribute
	sl = np.sort(data, order = 'sepal length')
	sw = np.sort(data, order = 'sepal width')
	pl = np.sort(data, order = 'petal length')
	pw = np.sort(data, order = 'petal width')
	attributes = np.array([], dtype={'names':('attr', 'threshold'),'formats': ('U20','f8')})
	#handling continous descriptive features
	for i in range(data.size-1):
		if sl[i]['class'] != sl[i+1]['class']:
			threshold = (sl[i]['sepal length'] + sl[i+1]['sepal length'])/2
			if np.array([('sepal length',threshold)], dtype=attributes.dtype) not in attributes:
				attributes = np.append(attributes,np.array([('sepal length',threshold)], dtype=attributes.dtype))
	
		if sw[i]['class'] != sw[i+1]['class']:
			threshold = (sw[i]['sepal width'] + sw[i+1]['sepal width'])/2
			if np.array([('sepal width',threshold)], dtype=attributes.dtype) not in attributes:
				attributes = np.append(attributes,np.array([('sepal width',threshold)], dtype=attributes.dtype))
		
		if pl[i]['class'] != pl[i+1]['class']:
			threshold = (pl[i]['petal length'] + pl[i+1]['petal length'])/2
			if np.array([('petal length',threshold)], dtype=attributes.dtype) not in attributes:
				attributes = np.append(attributes,np.array([('petal length',threshold)], dtype=attributes.dtype))

		if pw[i]['class'] != pw[i+1]['class']:
			threshold = (pw[i]['petal width'] + pw[i+1]['petal width'])/2
			if np.array([('petal width',threshold)], dtype=attributes.dtype) not in attributes:
				attributes = np.append(attributes,np.array([('petal width',threshold)], dtype=attributes.dtype))	
	return attributes

#calculate the entropy
def entropy(data):
	a = b = c = 0
	entropy = 0
	for i in range(data.size):
		if data[i]['class'] == "b'Iris-setosa'":
			a = a + 1
		elif data[i]['class'] == "b'Iris-versicolor'":
			b = b + 1
		else :  
			c = c + 1
	if a != 0:
		entropy -= (a/(a+b+c))*log2(a/(a+b+c))
	if b != 0:
		entropy -= (b/(a+b+c))*log2(b/(a+b+c))
	if c != 0:
		entropy -= (c/(a+b+c))*log2(c/(a+b+c))
	return entropy

#calculate the remainder
def rem(data,attr):
	rem = 0	
	data0 = data1 = np.array([],dtype = data.dtype)
	for i in range(data.size):
		if data[i][attr['attr']] < attr['threshold']:
			data0 = np.append(data0,data[i])
		else:
			data1 = np.append(data1,data[i])
	rem = entropy(data0)*data0.size/data.size + entropy(data1)*data1.size/data.size
	return rem

#calculate the information gain
def gain(data, attr):
	gain = entropy(data) - rem(data, attr)
	return gain

#choose the best attribute
def choose_best_attr(data, attributes):
	max_ig = best = 0
	for i in range(attributes.size):
		ig = gain(data, attributes[i])
		if ig > max_ig:
			max_ig = ig
			best = i
	return best

# get the most class in the dataset
def majority_value(data):
	a = b = c = 0
	for i in range(data.size):	
		if data[i]['class'] == "b'Iris-setosa'":
			a = a + 1
		elif data[i]['class'] == "b'Iris-versicolor'":
			b = b + 1
		else :  
			c = c + 1
	if a > b and a > c:
		return "b'Iris-setosa'"
	elif b > a and b > c:
		return "b'Iris-versicolor'"
	else: 
		return "b'Iris-virginica'"

#create decisiontree with attribute as label
def create_decision_tree(data, attributes, parent):
	target = data['class']
	if data.size == 0:
		return parent
	elif	attributes.size == 0:
		return majority_value(data)
	elif np.count_nonzero(target == target[0]) == target.size:
		return target[0]
	else:
		id = choose_best_attr(data,attributes)		
		best = attributes[id]
		tmp = np.delete(attributes,id)
		tree = {best['attr']:{}}
		data0 = data1 = np.array([],dtype = data.dtype)
		for i in range(data.size):
			if data[i][best['attr']] < best['threshold']:
				data0 = np.append(data0,data[i])
			else:
				data1 = np.append(data1,data[i])
		subtree = create_decision_tree(data0, tmp, majority_value(data))
		tree[best['attr']][0] = subtree
		subtree = create_decision_tree(data1, tmp, majority_value(data))
		tree[best['attr']][1] = subtree
	return tree

#create decisiontree with threshold as label
def create_decision_tree_ts(data, attributes, parent):
	target = data['class']
	if data.size == 0:
		return parent
	elif	attributes.size == 0:
		return majority_value(data)
	elif np.count_nonzero(target == target[0]) == target.size:
		return target[0]
	else:
		id = choose_best_attr(data,attributes)		
		best = attributes[id]
		tmp = np.delete(attributes,id)
		tree = {best['threshold']:{}}
		data0 = data1 = np.array([],dtype = data.dtype)
		for i in range(data.size):
			if data[i][best['attr']] < best['threshold']:
				data0 = np.append(data0,data[i])
			else:
				data1 = np.append(data1,data[i])
		subtree = create_decision_tree_ts(data0, tmp,majority_value(data))
		tree[best['threshold']][0] = subtree
		subtree = create_decision_tree_ts(data1, tmp,majority_value(data))
		tree[best['threshold']][1] = subtree
	return tree

def K_fold_cross_validation(data,k):	
	accuracy = 0
	precision_a = precision_b = precision_c = 0
	recall_a = recall_b = recall_c = 0
	tp_a = tp_b = tp_c = 0
	tn_a = tn_b = tn_c = 0
	fp_a = fp_b = fp_c = 0
	fn_a = fn_b = fn_c = 0
	for j in range(k):
		testing_data = np.split(data,k)[j]
		trainning_data = np.delete(data,np.s_[int(data.size/k*j):int(data.size/k*(j+1))])
		attributes = get_attributes(trainning_data)
		tree = create_decision_tree(trainning_data, attributes,0)
		tree_ts = create_decision_tree_ts(trainning_data, attributes,0)
		tp_a = tp_b = tp_c = 0
		tn_a = tn_b = tn_c = 0
		fp_a = fp_b = fp_c = 0
		fn_a = fn_b = fn_c = 0
		for i in range(testing_data.size):
			x = tree 
			y = tree_ts
			while(1):
				for attr in x.keys():
					for ts in y.keys():
						if testing_data[i][attr] < ts:
							x = x[attr][0]
							y = y[ts][0]
						else:	
							x = x[attr][1]
							y = y[ts][1]
				if (x == "b'Iris-setosa'" or x == "b'Iris-versicolor'" or x == "b'Iris-virginica'"):
					if x == "b'Iris-setosa'" and testing_data[i]['class'] == "b'Iris-setosa'":					
						tp_a = tp_a + 1
					elif x == "b'Iris-setosa'" and testing_data[i]['class'] != "b'Iris-setosa'":
						fp_a = fp_a + 1
					elif x != "b'Iris-setosa'" and testing_data[i]['class'] == "b'Iris-setosa'":
						fn_a = fn_a + 1
					else:	tn_a = tn_a + 1

					if x == "b'Iris-versicolor'" and testing_data[i]['class'] == "b'Iris-versicolor'":
						tp_b = tp_b + 1
					elif x == "b'Iris-versicolor'" and testing_data[i]['class'] != "b'Iris-versicolor'":
						fp_b = fp_b + 1
					elif x != "b'Iris-versicolor'" and testing_data[i]['class'] == "b'Iris-versicolor'":
						fn_b = fn_b + 1
					else:	tn_b = tn_b + 1

					if x == "b'Iris-virginica'" and testing_data[i]['class'] == "b'Iris-virginica'":
						tp_c = tp_c + 1
					elif x == "b'Iris-virginica'" and testing_data[i]['class'] != "b'Iris-virginica'":
						fp_c = fp_c + 1
					elif x != "b'Iris-virginica'" and testing_data[i]['class'] == "b'Iris-virginica'":
						fn_c = fn_c + 1
					else:	tn_c = tn_c + 1
					break
		accuracy += (tp_a + tp_b + tp_c)/testing_data.size
		precision_a += tp_a/(tp_a + fp_a)
		recall_a += tp_a/(tp_a + fn_a)
		precision_b += tp_b/(tp_b + fp_b)
		recall_b += tp_b/(tp_b + fn_b)
		precision_c += tp_c/(tp_c + fp_c)
		recall_c += tp_c/(tp_c + fn_c)			
	accuracy = accuracy/k
	precision_a = precision_a/k
	recall_a = recall_a/k
	precision_b = precision_b/k
	recall_b = recall_b/k
	precision_c = precision_c/k
	recall_c = recall_c/k
	acc = np.array([])
	acc = np.append(acc, accuracy)
	acc = np.append(acc, precision_a)
	acc = np.append(acc, recall_a)
	acc = np.append(acc, precision_b)
	acc = np.append(acc, recall_b)
	acc = np.append(acc, precision_c)
	acc = np.append(acc, recall_c)
	return acc

def random_forest(data,trees):
	best_acc = acc = 0
	for i in range(trees):
		np.random.shuffle(data)
		tmp = K_fold_cross_validation(data,5)
		if tmp[0]*1.5 + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]> best_acc:
			best_acc = tmp[0]*1.5 + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]
			acc = tmp
	
	print('{0:.3f}'.format(acc[0]))
	print('{0:.3f} {0:.3f}'.format(acc[1],acc[2]))
	print('{0:.3f} {0:.3f}'.format(acc[3],acc[4]))
	print('{0:.3f} {0:.3f}'.format(acc[5],acc[6]))
	return

random_forest(data,25)
