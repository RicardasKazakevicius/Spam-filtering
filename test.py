import time
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors

enronDirectory = 'Enron-data-set'
lingSpamDirectory = 'lingspam_public'
pu1Directory = 'PU1_Data_Set'

directory = enronDirectory

numberOfTests = 10
enron1 = 5172
enron2 = 5857
enron3 = 5512
enron4 = 6000
enron5 = 5175
enron6 = 6000
enronAll = 33716
lingSpam = 2893
pu1 = 1099

mostCommon = 500
noOfEmail = enronAll

file = open(str(noOfEmail) + ".txt", "w")

def make_Dictionary(root_dir):
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    all_words = []       
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d,f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    for line in m:
                        words = line.split()
                        all_words += words
    dictionary = Counter(all_words)
    
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(mostCommon)
    
    return dictionary
    
def extract_features(root_dir): 
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]  
    docID = 0
    features_matrix = np.zeros((noOfEmail,mostCommon))
    train_labels = np.zeros(noOfEmail)
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d,f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    all_words = []
                    for line in m:
                        words = line.split()
                        all_words += words
                    for word in all_words:
                      wordID = 0
                      for i,d in enumerate(dictionary):
                        if d[0] == word:
                          wordID = i
                          features_matrix[docID,wordID] = all_words.count(word)
                train_labels[docID] = int(mail.split(".")[-2] == 'spam')
                print mail.split(".")[-2] == 'spam'
                docID = docID + 1                
    return features_matrix,train_labels

def trainTest(model, name):
	accurasyList = [0]*numberOfTests
	for i in range(0, numberOfTests):
		X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.30)
		model.fit(X_train, y_train)
		result = model.predict(X_test)
		# print confusion_matrix(y_test, result)
		accurasy = accuracy_score(y_test, result)*100
		accurasyList[i] = accurasy
		#print accurasy
	file.write(name + "\n")
	file.write('{0:.2f}'.format(np.mean(np.array(accurasyList))) + "\n")
	file.write('{0:.2f}'.format(np.std(np.array(accurasyList))) + "\n")


root_dir = directory
dictionary = make_Dictionary(root_dir)

features_matrix, labels = extract_features(root_dir)

print (features_matrix.shape)
print (labels.shape)
print (sum(labels==0),sum(labels==1))

print "\nNN"
trainTest(MLPClassifier(), "NN")

print "\nLG"
trainTest(LogisticRegression(), "LG")

print "\nSVM"
trainTest(LinearSVC(), "SVM")

print "\nNB"
trainTest(MultinomialNB(), "NB")

print "\nTREE"
trainTest(DecisionTreeClassifier(), "TREE")

print "\nKNN"
trainTest(neighbors.KNeighborsClassifier(), "KNN")


file.close()