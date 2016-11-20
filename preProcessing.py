from gensim import corpora  ,models, similarities
from gensim.utils import smart_open, simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.classify.util
from nltk.stem.porter import *


sentence = """
The news followed signs on Thursday that the weight of the presidency is beginning to sink in for Trump, and that the President-elect may be shifting from the bomb-throwing tactics he employed during the campaign to a more nuanced approach."""




def tokenize(text):
	return [token for token in simple_preprocess(text) if token not in STOPWORDS]



def stem(tokens):
	stemmer = PorterStemmer()
	singles = [stemmer.stem(tok) for tok in tokens]
	return singles


def createSentence(textAsList):
	return' '.join(map(str, textAsList))



def sentimentAnalysis(text):
	sid = SentimentIntensityAnalyzer()
	ss = sid.polarity_scores(text)
	print ss



def processText(text):
	tokens = tokenize(text)
	# stems = stem(tokens)
	sentence = createSentence(tokens)
	sid = SentimentIntensityAnalyzer()
	ss = sid.polarity_scores(sentence)

	return ss

# {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}  raw
# {'neg': 0.151, 'neu': 0.849, 'pos': 0.0, 'compound': -0.4939}  tokenized
# {'neg': 0.151, 'neu': 0.849, 'pos': 0.0, 'compound': -0.4939} tokenized and stemmed



print processText(sentence)

