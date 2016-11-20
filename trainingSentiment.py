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
		result =  textClassification.feedData(data)
		print result
		# break
		


# def train(data):s
# 	print data
# 	articleDemScore = 0
# 	articleRepScore = 0
# 	threshold = 0.2
# 	count += 1 
# 	global repScore , demScore , count 
# 	for sentence , topic  in data:
# 		if len(topic) == 1:
# 			results = textClassification.classifySubjectSentiment(sentence, topic[0][1])
# 			if topic[0][1] == 'Democratic' and abs(results[1]['compound']) > threshold:
# 				demScore += results[1]['compound']
# 				articleDemScore += results[1]['compound']
# 			elif topic[0][1] == 'Republican' and abs(results[1]['compound']) > threshold:
# 				repScore += results[1]['compound']
# 				articleRepScore += results[1]['compound']




	# print '*****'
	# print demScore , repScore
	# if count == 200:
	# 	print repScore , demScore
	# 	sys.exit(0)




	# print results
	

getData('fox')



