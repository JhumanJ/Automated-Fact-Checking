"""

Utils function

"""

import json, re, string, time
from difflib import SequenceMatcher

"""
Save a dictionary to a json file.
"""


def saveDictToJson(dict, path):
    jsondump = json.dumps(dict)
    f = open(path, "w+")
    f.write(jsondump)
    f.close()

    return True


"""
Opens a json file to return a dict.
"""


def openJsonDict(path):
    start = time.time()
    with open(path) as handle:
        dictdump = json.loads(handle.read())
    print("File {} loaded in {} seconds.".format(path,time.time()-start))

    return dictdump


"""
Return a proximity ratio for two strings
"""


def textSimilarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


"""
Given a text return a list of words.
Keeps all alphanumeric string given a minimal length
"""


def splitWords(text, min_length=2):
    # Parse text in words. Split words using regex for words of minimal length 2
    words = re.sub('[' + string.punctuation + ']', '', text).split()
    return [word.lower() for word in words if len(word) >= min_length]


"""
Given a text returns a dictionary containing count of occurence of each word
"""


def wordCount(words):
    wordCountDict = {}
    for word in words:
        if word in wordCountDict:
            wordCountDict[word] += 1
        else:
            wordCountDict[word] = 1

    return wordCountDict


def removeStopWords(words):
    stopWords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
                 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did',
                 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
                 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself',
                 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', 'it', "it's",
                 'its', 'itself', "let's", 'me', 'more', 'most', 'my', 'myself', 'nor', 'of', 'on', 'once', 'only',
                 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', "she'd",
                 "she'll", "she's", 'should', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',
                 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
                 "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we',
                 "we'd", "we'll", "we're", "we've", 'were', 'what', "what's", 'when', "when's", 'where', "where's",
                 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', 'would', 'you', "you'd", "you'll",
                 "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

    return [word for word in words if word not in stopWords]
