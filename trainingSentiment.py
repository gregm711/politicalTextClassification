import textClassification

import sys
sys.path.append('../political-data/articles')
import iterData
demScore =  0
repScore = 0
count = 0

def getData(source):
	compoundScore = 0
	count
	generator = iterData.getData(source)
	while True:
		data = next(generator)
		train(data)


def train(data):
	articleDemScore = 0
	articleRepScore = 0
	threshold = 0.65
	global repScore , demScore , count 
	for sentence , topic  in data:
		if len(topic) == 1:
			results = textClassification.classifySubjectSentiment(sentence, topic[0][1])
			if topic[0][1] == 'Democratic' and abs(results[1]['compound']) > threshold:
				demScore += results[1]['compound']
				articleDemScore += results[1]['compound']
			elif topic[0][1] == 'Republican' and abs(results[1]['compound']) > threshold:
				repScore += results[1]['compound']
				articleDemScore += results[1]['compound']
			count += 1 

	print articleDemScore , articleRepScore
	if count == 200:
		print repScore , demScore
		sys.exit(0)




	# print results
	

print getData('fox')



