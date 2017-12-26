import numpy as np
import graphviz
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

#Iris
print('Iris')
#Load Data
iris = load_iris()
data = iris.data
target = iris.target
random_array = np.arange(data.shape[0])
np.random.shuffle(random_array)
data = data[random_array]
target = target[random_array]
train_data = np.split(data, [(int)(len(data)*0.7)])[0]
train_target = np.split(target, [(int)(len(target)*0.7)])[0]
test_data = np.split(data, [(int)(len(data)*0.7)])[1]
test_target = np.split(target, [(int)(len(target)*0.7)])[1]

#Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

#Plot the Decision Tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")

#Print the accuracy
accracy = clf.score(test_data,test_target)
print('DecisionTree accuracy:', end = "\t")
print(accracy)

#KNN
neigh = KNeighborsClassifier(n_neighbors=5)
accracy = neigh.fit(train_data, train_target).score(train_data, train_target)
print('K = 5, KNN accuracy:', end = "\t")
print(accracy)

#Naive Bayes
gnb = GaussianNB()
accracy = gnb.fit(train_data, train_target).score(test_data,test_target)
PDF = gnb.predict_proba(test_data)
print('Naive Bayes accuracy:', end = "\t")
print(accracy)
print('The probability of the samples for each target class.')
print(PDF)

#Forestfires
print('Forestfires')
#Load Data
df = pd.read_csv('forestfires.csv')
feature_names = list(df)[:12]
target_names = ['0','1','2','3','4','5']
map_month2int = {'oct': 10, 'sep': 9, 'aug': 8, 'jul': 7, 'feb': 2, 'jun': 6, 'nov': 11, 'apr': 4, 'mar': 3, 'may': 5, 'jan': 1, 'dec': 12}
map_day2int = {'fri': 5, 'sat': 6, 'mon': 1, 'wed': 3, 'tue': 2, 'thu': 4, 'sun': 7}
df['month'] = df['month'].replace(map_month2int)
df['day'] = df['day'].replace(map_day2int)
npdata =df.values
np.random.shuffle(npdata)
data = npdata[:,0:12]
target = npdata[:,12:13].flatten()

for i in range(len(target)):
	if target[i] == 0:
		target[i] = 0
	elif target[i] < 1:
		target[i] = 1
	elif target[i] < 10:
		target[i] = 2
	elif target[i] < 100:
		target[i] = 3
	elif target[i] < 1000:
		target[i] = 4
	else:
		target[i] = 5

train_data = np.split(data, [(int)(len(data)*0.7)])[0]
train_target = np.split(target, [(int)(len(target)*0.7)])[0]
test_data = np.split(data, [(int)(len(data)*0.7)])[1]
test_target = np.split(target, [(int)(len(target)*0.7)])[1]

#Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

#Plot the Decision Tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=feature_names,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("forestfires")

#Print the accuracy
accracy = clf.score(test_data,test_target)
print('DecisionTree accuracy:', end = "\t")
print(accracy)

#KNN
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_data, train_target)
accracy = neigh.score(train_data, train_target)
print('K = 5, KNN accuracy:', end = "\t")
print(accracy)

#Naive Bayes
cat_train_data = train_data[:,:4]
cot_train_data = train_data[:,4:]
cat_test_data = test_data[:,:4]
cot_test_data = test_data[:,4:]

#Laplace smooth
mnb = MultinomialNB()
PDF_cat = mnb.fit(cat_train_data, train_target).predict_proba(cat_test_data)

gnb = GaussianNB()
PDF_cot = gnb.fit(cot_train_data, train_target).predict_proba(cot_test_data)


mul = np.multiply(PDF_cat,PDF_cot)
correct = 0
for i in range(len(test_target)):
	if np.argmax(mul[i]) == test_target[i]:
		correct = correct + 1
accuracy = correct/len(test_target)
print('Naive Bayes accuracy:', end = "\t")
print(accuracy)
print('The probability of the samples for each target class.')
print(PDF_cot)
