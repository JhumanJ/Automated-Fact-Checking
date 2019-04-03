
from lib.datasets_parsing import *
from lib.stats import *
from paths import *
from tqdm import tqdm
from heapq import heappush, heappushpop
from q1 import getTextStatistics

import os, json, gc

"""
Uses drichlet to compute 5 most relevant document for each claims.
To speed up computations pre-filters documents to keep only relevant ones.

To do so, we use the inverted index
"""
def findRelevantDocumentsClaims(nbClaims=10000):

    relevantDocsPath = cache_path + "relevantDocs.json"
    docInvertedIndexFile = output_path + 'docInvertedIndex.json'

    # If already computed just return it
    if os.path.isfile(relevantDocsPath):
        print("Computations already done. Loading results from: {}", relevantDocsPath)
        return openJsonDict(relevantDocsPath)

    # First load claims
    claims = load_dataset_json(train_path)[0:nbClaims]
    print("{} claims loaded,".format(nbClaims))

    # Loading inverted index
    print("Loading inverted index.")
    invertedIndex = openJsonDict(docInvertedIndexFile)
    print("Inverted index loaded. Length: {}".format(len(invertedIndex)))

    relevantDocs = {}
    for claim in tqdm(claims):
        docs = {}
        words = set(removeStopWords(splitWords(claim['claim'])))

        # For each word, find candidate docs in the inverted index
        for word in words:
            if not word in invertedIndex:
                print('\033[1;31;40m /!\ Error word '+word+ ' not found in inverted index. Continuing. \033[1;37;40m')
                continue

            for (docId, count) in invertedIndex[word]:
                if docId in docs:
                    docs[docId] += 1
                else:
                    docs[docId] = 1

        # Now only keep the docs with at least two different common words
        relevantDocs[claim['id']] = []
        for key in docs:
            # If in a given doc we have more than 2 different words in common
            if docs[key] >= 2:
                relevantDocs[claim['id']].append(key)
        print(len( relevantDocs[claim['id']]))

    # Now save relevant docs to json
    del invertedIndex
    gc.collect()
    saveDictToJson(relevantDocs,relevantDocsPath)
    return relevantDocs

def question4():

    findRelevantDocumentsClaims()
    return

    # TODO: remove return above. below is for later when computing idf Below is

    # Compute vocabulary size and collection frequency (using result from q1)
    wordsDictionnary = getTextStatistics()
    vocSize = len(wordsDictionnary.keys())
    collectionFrequency = sum(wordsDictionnary.values())
    avgWordPerDocument = float(collectionFrequency) / len(wikiArticles)
    print(
        "Vocabulary size: {}, collection frequency: {}, avg word per document: {}".format(vocSize, collectionFrequency,
                                                                                          avgWordPerDocument))


question4()
