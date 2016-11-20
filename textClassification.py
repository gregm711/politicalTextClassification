# -*- coding: utf-8 -*- 
import os
import random
import csv
import time
import pandas as pd
import numpy as np
import codecs
import pickle
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from gensim import corpora  ,models, similarities
from gensim.utils import smart_open, simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.classify.util
from nltk.stem.porter import *




"""
RESULTS:
Count Vectorizer:

	Text body:
		raw       =  0.788218793829 , 0.788218793829
		tokenized =  0.793828892006 , 0.788218793829 ,  0.783309957924
		alpha 1 :
			stemmed and tokenizes = 0.812762973352 , 80.1
		alpha 0.5:
			stemmed and tokenizes = 0.81206171108 , 
		alpha = 0.2:
			stemmed and tokenizes = 0.82206171108 , 
		alpha = 0.1
			stemmed and tokenizes = 0.834502103787 , 


	sentences:

		Stemmed and tokenizez 0.875553680658



TFidf Vectorizer:
Stemmed And Tokenized:
	use_idf=False:
		0.836743303101
	use_idf=True:
		alpha  = 1
		0.89791183294
		alha = 0.5:
			0.909583069676
		0.93004288828

"""
def tokenize(text):
	return [token for token in simple_preprocess(text) if token not in STOPWORDS]


def stem(tokens):
	stemmer = PorterStemmer()
	singles = [stemmer.stem(tok) for tok in tokens]
	return singles


def testClassifier(testObjects, testTargets, classifier, vectorizer):


	exampleCounts = vectorizer.transform(testObjects)
	predictions = classifier.predict(exampleCounts)
	print np.mean(predictions == testTargets)



def createSentence(textAsList):
	return' '.join(map(str, textAsList))

# def createClassifier():
# 	text = []
# 	targets = []	
# 	count = 0
# 	together  = []
# 	t1 = time.time()
# 	with open('formattedDebate.csv', 'r') as csvFile:
# 		reader = csv.reader(csvFile)
# 		for row in reader:
# 			count  += 1
# 			try:
# 				encoded = row[1].decode('utf-8','ignore').encode('utf-8', 'ignore')
# 				sentences = encoded.split('.')
# 				for sentence in sentences:
# 					tokens = tokenize(sentence)
# 					stems = stem(tokens)
# 					newSent = createSentence(tokens)
# 					together.append((newSent, int(row[0])))
# 			except Exception as e :
# 				print e 
# 		print 'done' , time.time() - t1 , count

# 		random.shuffle(together)
		# trainObjects  =   [sentence for (sentence , val) in together[:int(len(together) * 0.8)]]
		# trainTargets  =   [val for (sent, val) in together[:int(len(together) * 0.8)]]
		# # trainObjects  =   [sentence for (sentence , val) in together]
		# # trainTargets  =   [val for (sent, val) in together]
		# testObjects   =   [sentence for (sentence , val) in together[:int(len(together) * .2)]]
		# testTargets   =   [val for (sentence , val) in together[:int(len(together) * .2)]]
			
# 		# vectorizer = TfidfVectorizer()
# 		vectorizer = CountVectorizer()
# 		counts = vectorizer.fit_transform(trainObjects)
# 		classifier = MultinomialNB(alpha = 0.5)
# 		classifier.fit(counts, trainTargets)

# 		testClassifier(testObjects, testTargets, classifier, vectorizer)

# 		classifierFile = open('topicsClassifierStemmedTokenized.pkl', 'w')
# 		vectorizerFile = open('topicsVectorizerStemmedTokenized.pkl', 'w')
# 		pickle.dump(classifier, classifierFile)
# 		pickle.dump(vectorizer, vectorizerFile)
# 		print 'done'


"""
abortion = folder + 'abortion'
 	creation = folder  + 'creation'
 	gayRights = folder + 'gayRights'
 	god = folder +  'god'
 	guns = folder  + 'guns'
 	healthcare = folder + 'healthcare'
"""

def createIndividuleClassifier(topic, sentencesBool):
	text = []
	targets = []	
	count = 0
	together  = []
	t1 = time.time()
	with open('formatted'+ topic + '.csv', 'r') as csvFile:
		reader = csv.reader(csvFile)
		for row in reader:
			count  += 1
			try:
				encoded = row[1].decode('utf-8','ignore').encode('utf-8', 'ignore')
				sentences = encoded.split('.')
				if sentencesBool == 'sentences':
					for sentence in sentences:
						tokens = tokenize(sentence)
						stems = stem(tokens)
						newSent = createSentence(tokens)
						together.append((newSent, int(row[0])))
				if sentencesBool != 'sentences':
					tokens = tokenize(encoded)
					stems = stem(tokens)
					newSent = createSentence(tokens)
					together.append((newSent, int(row[0])))
			except Exception as e:
				print e 
		random.shuffle(together)
		trainObjects  =   [sentence for (sentence , val) in together[:int(len(together) * 0.8)]]
		trainTargets  =   [val for (sent, val) in together[:int(len(together) * 0.8)]]
		# trainObjects  =   [sentence for (sentence , val) in together]
		# trainTargets  =   [val for (sent, val) in together]
		testObjects   =   [sentence for (sentence , val) in together[:int(len(together) * .2)]]
		testTargets   =   [val for (sentence , val) in together[:int(len(together) * .2)]]
			
		# vectorizer = TfidfVectorizer()
		print testTargets.count(1)
		print testTargets.count(2)
		print trainTargets.count(1)
		print trainTargets.count(2)
		vectorizer = TfidfVectorizer()
		counts = vectorizer.fit_transform(trainObjects)
		classifier = MultinomialNB(alpha = 0.5)
		classifier.fit(counts, trainTargets)

		testClassifier(testObjects, testTargets, classifier, vectorizer)

		classifierFile = open('topicsClassifierStemmedTokenized' + topic  + sentencesBool + '.pkl', 'w')
		vectorizerFile = open('topicsVectorizerStemmedTokenized' + topic + sentencesBool + '.pkl', 'w')
		pickle.dump(classifier, classifierFile)
		pickle.dump(vectorizer, vectorizerFile)
		print 'done'


# createIndividuleClassifier('Debate', '')


def processArray(textArray):
	result = []
	for text in textArray:
		tokens = tokenize(text)
		stems = stem(tokens)
		sentence = createSentence(stems)
		result.append(sentence)
	return result


def classifyText(text, topic, sentencesBool):
	classifierFile = open('topicsClassifierStemmedTokenized' + topic + sentencesBool +  '.pkl', 'rb')
	vectorizerFile = open('topicsVectorizerStemmedTokenized' + topic + sentencesBool +'.pkl', 'rb')
	if sentencesBool == 'sentences':
		sentences = text[0].split('.')
		finalText  = processArray(sentences)
	elif sentencesBool != 'sentences':
		finalText = processArray(text)
	classifier = pickle.load(classifierFile)
	vectorizer = pickle.load(vectorizerFile)
	finalCounts = vectorizer.transform(finalText)
	confidence = classifier.predict_proba(finalCounts)
	prediction = classifier.predict(finalCounts)

	return prediction , confidence

examples = ['Science says it\'s not a baby.I beg to differ.Science said it is a baby.The baby has its own DNA and at nine weeks he has a heartbeat.A baby at twelve weeks is very']


print classifyText(examples, 'abortion', '')

# def testClassifier():


# 	classifierFile = open('topicsClassifierStemmedTokenized.pkl', 'rb')
# 	vectorizerFile = open('topicsVectorizerStemmedTokenized.pkl', 'rb')
# 	vectorizer = pickle.load(vectorizerFile)
# 	classifier = pickle.load(classifierFile)
# 	examples = ['Science says it\'s not a baby.I beg to differ.Science said it is a baby.The baby has its own DNA and at nine weeks he has a heartbeat.A baby at twelve weeks is very']
	
# 	example_counts = vectorizer.transform(examples)
# 	predictions = classifier.predict(example_counts)
# 	predictions
# 	print predictions

# # testClassifier()

# print classifyText(examples)






# 