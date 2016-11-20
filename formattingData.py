import csv
import cPickle as pickle
import os

"""
guns: 2  dem
abortion: 1 = dem
creation: 2 = dem
gayRights: 1 = dem
God: 2  =dem
healthcare: 1 = dem
"""
def saveText():
	folder = '/Users/gregmiller/Desktop/SomasundaranWiebe-politicalDebates/'
	topic = 'guns'
	abortion = folder + 'abortion'
	creation = folder  + 'creation'
	gayRights = folder + 'gayRights'
	god = folder +  'god'
	guns = folder  + 'guns'
	healthcare = folder + 'healthcare'
	path  = guns
	with open('formatted' + topic + '.csv', 'a') as smallFile:
		with open('formattedDebate.csv', 'a') as csvFile:
			for filename in os.listdir(path):
				with open(path + '/'+ filename, 'r') as f:
					count = 0
					data = ''
					for line in f :
						count += 1 
						if count == 1:
							stance = int(line[len('#stance=stance')])
							if topic == 'creation' or topic == 'god' or topic == 'guns':
								if stance == 2:
									stance = 1
								elif stance ==1:
									stance = 2
						if count > 3:
							data = data + line
					fields = [stance, data]
					writer = csv.writer(csvFile)
					newWriter  = csv.writer(smallFile)

					writer.writerow(fields)
					newWriter.writerow(fields)



# saveText()


# import collections
# import nltk.metrics
# from nltk.classify import NaiveBayesClassifier
# from nltk.corpus import movie_reviews
 
# def word_feats(words):
# 	return dict([(word, True) for word in words])
 
# negids = movie_reviews.fileids('neg')
# posids = movie_reviews.fileids('pos')

# negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
# posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
 
# negcutoff = len(negfeats)*3/4
# poscutoff = len(posfeats)*3/4
 
# trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
# testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
# print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
# classifier = NaiveBayesClassifier.train(trainfeats)
# refsets = collections.defaultdict(set)
# testsets = collections.defaultdict(set)

 
# for i, (feats, label) in enumerate(testfeats):
# 	refsets[label].add(i)
# 	observed = classifier.classify(feats)
# 	testsets[observed].add(i)
 
# classifierFile = open('semanticClassifier.pkl', 'w')
# pickle.dump(classifier, classifierFile,protocol=2)
# sentance = ['great', 'movie']
# feats = word_feats(sentance)
# print 'here'
# classifierFile = open('semanticClassifier.pkl', 'rb')

# classifier = pickle.load(classifierFile)
# print 'here'
# prediction = classifier.classify(feats)
# print prediction
# # print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
# # print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
# # print 'pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
# # print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
# # print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
# print 'neg F-measure:', nltk.metrics.f_mea
