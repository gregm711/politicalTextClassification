import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from gensim.parsing.preprocessing import STOPWORDS
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from gensim.utils import smart_open, simple_preprocess
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

def evaluate_classifier(pos, neg):
    together = []
    for sentence , val in pos:
        tokens = tokenize(sentence)  
        newSent = createSentence(tokens) 
        vals = (newSent, 'pos')
        together.append(vals)
    for sentence, val in neg:

        tokens = tokenize(sentence)
        newSent = createSentence(tokens)   
        vals = (newSent, 'neg') 
        together.append(vals)
    trainObjects  =   [sentence for (sentence , val) in together[:int(len(together) * 0.8)]]
    trainTargets  =   [val for (sent, val) in together[:int(len(together) * 0.8)]]
    testObjects   =   [sentence for (sentence , val) in together[:int(len(together) * .2)]]
    testTargets   =   [val for (sentence , val) in together[:int(len(together) * .2)]]
    vectorizer = TfidfVectorizer()
    counts = vectorizer.fit_transform(trainObjects)
    classifier = MultinomialNB(alpha = 1)
    classifier.fit(counts, trainTargets)
    # testClassifier(testObjects, testTargets, classifier, vectorizer)
    classifierFile = open('sentenceSentimentClassifer.pkl', 'w')
    vectorizerFile = open('sentenceVectorizer.pkl', 'w')
    pickle.dump(classifier, classifierFile)
    pickle.dump(vectorizer, vectorizerFile)





    # negids = movie_reviews.fileids('neg')
    # posids = movie_reviews.fileids('pos')
 
    # negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    # posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
    

    # negcutoff = len(negfeats)*3/4
    # poscutoff = len(posfeats)*3/4
 
    # trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    # testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
 
    # classifier = NaiveBayesClassifier.train(trainfeats)
    # refsets = collections.defaultdict(set)
    # testsets = collections.defaultdict(set)
 
    # for i, (feats, label) in enumerate(testfeats):
    #         refsets[label].add(i)
    #         observed = classifier.classify(feats)
    #         testsets[observed].add(i)
 
    # print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    # # print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
    # # print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
    # # print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
    # # print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
    # classifier.show_most_informative_features()


def processArray(textArray):
    result = []
    for text in textArray:
        tokens = tokenize(text)

        sentence = createSentence(tokens)
        result.append(sentence)
    return result


def classify(text):
    classifierFile = open('sentenceSentimentClassifer.pkl', 'rb')
    vectorizerFile = open('sentenceVectorizer.pkl', 'rb')
    classifier = pickle.load(classifierFile)
    vectorizer = pickle.load(vectorizerFile)
    finalText  = processArray(text)
    finalCounts = vectorizer.transform(finalText)
    confidence = classifier.predict_proba(finalCounts)
    prediction = classifier.predict(finalCounts)
    return prediction , confidence , finalText





 
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    pass
    # bigram_finder = BigramCollocationFinder.from_words(words)
    # bigrams = bigram_finder.nbest(score_fn, n)
    # return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
 

def createSentence(textAsList):
    return' '.join(map(str, textAsList))
def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]


testSentences = ["Dan, what about this tactic to go to the left of Hillary Clinton on her hawkishness, on the fact that she's trigger happy, the fact that you have a Republican nominee now saying that the Democrat is more likely to be engaged in countries around the world and we shouldn't be "]

print classify(testSentences)

# # The news followed signs on Thursday that the weight of the presidency is beginning to sink in for Trump, and that the President-elect may be shifting from the bomb-throwing tactics he employed during the campaign to a more nuanced approach."""

def stem(tokens):
    stemmer = PorterStemmer()
    singles = [stemmer.stem(tok) for tok in tokens]
    return singles

def loadData():
    neg = []
    pos = []
    with open('/Users/gregmiller/Downloads/rt-polaritydata/rt-polaritydata/rt-polarity.neg') as negFile:
        for line in negFile:
            tmp = (line,'neg')
            neg.append(tmp)

        with open('/Users/gregmiller/Downloads/rt-polaritydata/rt-polaritydata/rt-polarity.pos') as posFile:
            for line in negFile:
                tmp = (line,'pos')
                pos.append(tmp)
            return pos , neg

# pos, neg = loadData()
# evaluate_classifier(pos, neg)

# evaluate_classifier(bigram_word_feats)

# def word_feats(words):
#     return dict([(word, True) for word in words])
 
# evaluate_classifier(word_feats)