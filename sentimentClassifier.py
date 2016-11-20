from gensim import corpora  ,models, similarities
from gensim.utils import smart_open, simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.classify.util
from nltk.stem.porter import *

def tokenize(text):
		return [token for token in simple_preprocess(text) if token not in STOPWORDS]


sentence = """
The news followed signs on Thursday that the weight of the presidency is beginning to sink in for Trump, and that the President-elect may be shifting from the bomb-throwing tactics he employed during the campaign to a more nuanced approach."""



tokens =  tokenize(sentence)
stemmer = PorterStemmer()

singles = [stemmer.stem(tok) for tok in tokens]
newSent = ' '.join(map(str, tokens))
print newSent
sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(newSent)

print ss


