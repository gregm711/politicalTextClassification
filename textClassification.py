# -*- coding: utf-8 -*- 
import os
import random
import csv
import pandas as pd
import numpy
import codecs
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def trainClassifier():
	text = []
	targets = []	
	fails = 0
	with open('formattedDebate.csv', 'r') as csvFile:
		reader = csv.reader(csvFile)
		for row in reader:
			try:
				text.append(row[1].decode('utf-8','ignore').encode('utf-8', 'ignore'))
				targets.append(int(row[0]))
			except Exception as e:
				print row
				print e 

				break

			
		vectorizer = CountVectorizer()
		counts = vectorizer.fit_transform(text)
		classifier = MultinomialNB()
		classifier.fit(counts, targets)

		classifierFile = open('topicsClassifier.pkl', 'w')
		vectorizerFile = open('topicsVectorizer.pkl', 'w')
		pickle.dump(classifier, classifierFile)
		pickle.dump(vectorizer, vectorizerFile)


def testClassifier():

	classifierFile = open('topicsClassifier.pkl', 'rb')
	vectorizerFile = open('topicsVectorizer.pkl', 'rb')
	vectorizer = pickle.load(vectorizerFile)
	classifier = pickle.load(classifierFile)
	examples = ['Science says it\'s not a baby.I beg to differ.Science said it is a baby.The baby has its own DNA and at nine weeks he has a heartbeat.A baby at twelve weeks is very']
	
	example_counts = vectorizer.transform(examples)
	predictions = classifier.predict(example_counts)
	predictions
	print predictions

testClassifier()























def saveText():
	folder = '/Users/gregmiller/Desktop/SomasundaranWiebe-politicalDebates/'
	abortion = folder + 'abortion'
	creation = folder  + 'creation'
	gayRights = folder + 'gayRights'
	god = folder +  'god'
	guns = folder  + 'guns'
	healthcare = folder + 'healthcare'
	path  = healthcare
	with open('formattedDebate.csv', 'a') as csvFile:
		for filename in os.listdir(path):
			with open(path + '/'+ filename, 'r') as f:
				count = 0
				data = ''
				for line in f :
					count += 1 
					if count == 1:
						stance = line[len('#stance=stance')]
					if count > 3:
						data = data + line
				fields = [stance, data]
				writer = csv.writer(csvFile)
				writer.writerow(fields)



# saveText()


# saveText()