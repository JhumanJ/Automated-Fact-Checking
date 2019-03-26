"""

Stats functions

"""
import math
import numpy as np
from lib.utils import *
from tqdm import tqdm

"""
Given a text, returns a dictionnary with tf score for each words
"""


def computeTf(words):
    nbWords = len(words)
    wordCounts = wordCount(words)

    for key in wordCounts:
        wordCounts[key] = float(wordCounts[key]) / nbWords

    return wordCounts


"""
Given a text, returns a dictionnary with idf score for each words

idToIgnore parameter used to avoid computing IDF over document itself
"""


def computeIDF(words, invertedIndex, documentCount, docIdf=None):
    # Only unique word matters
    words = set(words)

    # Compute IDF for each unique words
    score = {}

    if not docIdf is None:
        for word in words:
            score[word] = docIdf[word]
    else:
        for word in words:
            score[word] = math.log10(float(documentCount) / len(invertedIndex[word]))

    return score


"""
Compute td idf
Can ignore document id
"""


def computeTfIdf(text, id, invertedIndex, documentCount, docIdf=None):
    words = splitWords(text)

    # todo: optimize idf computations

    tf = computeTf(words)
    idf = computeIDF(words, invertedIndex, documentCount, docIdf)

    wordDic = {}

    # Compute each tfidf
    for key in tf:
        wordDic[key] = tf[key] * idf[key]

    return {
        'id': id,
        'text': text,
        'tfidfs': wordDic
    }


"""
Computed tf-idf for a set of claims.
"""


def computeTfIdfForClaims(claims, invertedIndex, documentCount):
    tfIDfClaims = []

    for claim in tqdm(claims):
        tfIDfClaims.append(computeTfIdf(claim['claim'], claim['id'], invertedIndex, documentCount))

    return tfIDfClaims


"""
Given two tf-idf dict, return the cosin similiarity
"""


def cosineSim(a, b):
    words = set(list(a.keys()) + list(b.keys()))
    a = [(a[word] if word in a else 0) for word in words]
    b = [(b[word] if word in b else 0) for word in words]

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))


"""
Given a document and a set of words (possibly used in a query), return 
the query likelyhood model

So this is without smoothing!
"""


def computeQueryLikelihoodModel(docWords, queryWords):
    nbWords = len(docWords)
    wordCounts = wordCount(docWords)

    model = {}
    for key in queryWords:
        if not key in wordCounts:
            model[key] = 0
        else:
            model[key] = float(wordCounts[key]) / nbWords

    return model


"""
Same as above with Laplace smoothing.
"""
def computeLaplaceQueryLikelihoodModel(docWords, queryWords, vocSize):
    nbWords = len(docWords)
    wordCounts = wordCount(docWords)

    model = {}

    # Precompute default value for a missing word
    defaultVal = float(1.0) / (nbWords + vocSize)

    for key in queryWords:
        if not key in wordCounts:
            model[key] = defaultVal
        else:
            model[key] = float(wordCounts[key] + 1) / (nbWords + vocSize)

    return model


"""
Compute query score for a given model and a given query
query has to be a list of words
"""


def computeQueryScore(model, query):
    score = 1.0
    for word in query:
        if model[word] == 0:
            return 0
        else:
            score = score * model[word]

    return score
