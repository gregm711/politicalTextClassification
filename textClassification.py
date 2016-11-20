# -*- coding: utf-8 -*- 
import os
import random
import re
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

abortion = folder + 'abortion'
 	creation = folder  + 'creation'
 	gayRights = folder + 'gayRights'
 	god = folder +  'god'
 	guns = folder  + 'guns'
 	healthcare = folder + 'healthcare'


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
		testObjects   =   [sentence for (sentence , val) in together[:int(len(together) * .2)]]
		testTargets   =   [val for (sentence , val) in together[:int(len(together) * .2)]]
		vectorizer = TfidfVectorizer()
		counts = vectorizer.fit_transform(trainObjects)
		classifier = MultinomialNB(alpha = 0.5)
		classifier.fit(counts, trainTargets)
		testClassifier(testObjects, testTargets, classifier, vectorizer)
		classifierFile = open('topicsClassifierStemmedTokenized' + topic  + sentencesBool + '.pkl', 'w')
		vectorizerFile = open('topicsVectorizerStemmedTokenized' + topic + sentencesBool + '.pkl', 'w')
		pickle.dump(classifier, classifierFile)
		pickle.dump(vectorizer, vectorizerFile)


# createIndividuleClassifier('Debate', '')


def processArray(textArray):
	result = []
	for text in textArray:
		tokens = tokenize(text)
		stems = stem(tokens)
		sentence = createSentence(stems)
		result.append(sentence)
	return result


def topicClassifier(text, topic, sentencesTag):
	classifierFile = open('topicsClassifierStemmedTokenized' + topic + sentencesTag +  '.pkl', 'rb')
	vectorizerFile = open('topicsVectorizerStemmedTokenized' + topic + sentencesTag +'.pkl', 'rb')
	if sentencesTag == 'sentences':
		sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text[0])
		finalText  = processArray(sentences)
	elif sentencesTag != 'sentences':
		finalText = processArray(text)
	classifier = pickle.load(classifierFile)
	vectorizer = pickle.load(vectorizerFile)
	finalCounts = vectorizer.transform(finalText)
	confidence = classifier.predict_proba(finalCounts)
	prediction = classifier.predict(finalCounts)
	return prediction , confidence , finalText


def classifyTopic(examples, topic):
	confidenceThreshold = 0.6
	usingSentences , sentencesConfidence , finalText = topicClassifier(examples, topic, 'sentences')
	results = []
	for x in  xrange(len(sentencesConfidence) - 1):
		maxVal = max(sentencesConfidence[x][0], sentencesConfidence[x][1])
		if maxVal >= confidenceThreshold:
			confidence  = sentencesConfidence[x]
			tmpTup = (usingSentences[x], finalText[x], maxVal)
			results.append(tmpTup)
	noSentences    , noSentencesConfidence, finalText = topicClassifier(examples, topic, '')
	maxVal = max(noSentencesConfidence[0][0], noSentencesConfidence[0][1])
	if maxVal >= confidenceThreshold:
		tmpTup = (noSentences[0], finalText[0], maxVal)
		results.append(tmpTup)
	return results





def extractSubject(text):
	token = nltk.word_tokenize(text)
	return nltk.pos_tag(token)



# sentence = 'I shot an elephant in my pajamas'
# tokens = tokenize(sentence)
# stemmer = PorterStemmer()
# singles = [stemmer.stem(tok) for tok in tokens]
# print extractSubject(sentence)

def classifySubjectSentiment(sentence, subject):
	tokens =  tokenize(sentence)
	stemmer = PorterStemmer()
	singles = [stemmer.stem(tok) for tok in tokens]
	newSent = ' '.join(map(str, singles))
	sid = SentimentIntensityAnalyzer()
	tokenizedPolarity = sid.polarity_scores(newSent)
	rawPolarity = sid.polarity_scores(sentence)
	

	return rawPolarity, tokenizedPolarity , subject

# testSentences = ["BAIER: Mercedes, Hillary Clinton going after again and again and again temperament of Donald Trump"]
# # sentence = """

# # The news followed signs on Thursday that the weight of the presidency is beginning to sink in for Trump, and that the President-elect may be shifting from the bomb-throwing tactics he employed during the campaign to a more nuanced approach."""

# # trump = ["Trump's remarks triggered a barrage of posts on Twitter, most of which were critical of the president-elect."]
# for sentence in testSentences:
# 	print classifySubjectSentiment(sentence, 'trump')






"""

({'neg': 0.264, 'neu': 0.64, 'pos': 0.096, 'compound': -0.4754}, {'neg': 0.529, 'neu': 0.471, 'pos': 0.0, 'compound': -0.5423})
({'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.6249}, {'neg': 0.0, 'neu': 0.196, 'pos': 0.804, 'compound': 0.6249})
({'neg': 0.0, 'neu': 0.406, 'pos': 0.594, 'compound': 0.6588}, {'neg': 0.0, 'neu': 0.196, 'pos': 0.804, 'compound': 0.6249})
({'neg': 0.6, 'neu': 0.4, 'pos': 0.0, 'compound': -0.4588}, {'neg': 0.744, 'neu': 0.256, 'pos': 0.0, 'compound': -0.4404})
({'neg': 0.592, 'neu': 0.408, 'pos': 0.0, 'compound': -0.4404}, {'neg': 0.649, 'neu': 0.351, 'pos': 0.0, 'compound': -0.5719})
({'neg': 0.273, 'neu': 0.413, 'pos': 0.314, 'compound': 0.128}, {'neg': 0.452, 'neu': 0.548, 'pos': 0.0, 'compound': -0.5106})



({'neg': 0.264, 'neu': 0.64, 'pos': 0.096, 'compound': -0.4754}, {'neg': 0.366, 'neu': 0.488, 'pos': 0.146, 'compound': -0.4215})
({'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.6249}, {'neg': 0.0, 'neu': 0.196, 'pos': 0.804, 'compound': 0.6249})
({'neg': 0.0, 'neu': 0.406, 'pos': 0.594, 'compound': 0.6588}, {'neg': 0.0, 'neu': 0.196, 'pos': 0.804, 'compound': 0.6249})
({'neg': 0.6, 'neu': 0.4, 'pos': 0.0, 'compound': -0.4588}, {'neg': 0.75, 'neu': 0.25, 'pos': 0.0, 'compound': -0.4588})
({'neg': 0.592, 'neu': 0.408, 'pos': 0.0, 'compound': -0.4404}, {'neg': 0.592, 'neu': 0.408, 'pos': 0.0, 'compound': -0.4404})
({'neg': 0.273, 'neu': 0.413, 'pos': 0.314, 'compound': 0.128}, {'neg': 0.327, 'neu': 0.297, 'pos': 0.376, 'compound': 0.128})








Not tokenizing or stemming
{'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.6249}
{'neg': 0.0, 'neu': 0.406, 'pos': 0.594, 'compound': 0.6588}
{'neg': 0.6, 'neu': 0.4, 'pos': 0.0, 'compound': -0.4588}
{'neg': 0.592, 'neu': 0.408, 'pos': 0.0, 'compound': -0.4404}
{'neg': 0.273, 'neu': 0.413, 'pos': 0.314, 'compound': 0.128}


with stemming:
{'neg': 0.0, 'neu': 0.196, 'pos': 0.804, 'compound': 0.6249}
{'neg': 0.0, 'neu': 0.196, 'pos': 0.804, 'compound': 0.6249}
{'neg': 0.75, 'neu': 0.25, 'pos': 0.0, 'compound': -0.4588}
{'neg': 0.592, 'neu': 0.408, 'pos': 0.0, 'compound': -0.4404}
{'neg': 0.327, 'neu': 0.297, 'pos': 0.376, 'compound': 0.128}
"""



# 