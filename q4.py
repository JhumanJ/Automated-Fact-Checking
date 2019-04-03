
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
def findRelevantDocumentsClaims(claims):

    relevantDocsPath = cache_path + "relevantDocs"+str(len(claims))+"Claims.json"
    docInvertedIndexFile = output_path + 'docInvertedIndex.json'

    # If already computed just return it
    if os.path.isfile(relevantDocsPath):
        print("Computations already done. Loading results from: ", relevantDocsPath)
        return openJsonDict(relevantDocsPath)

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

def computeTop5dirichlet(claims,relevantDocs):
    claimsTop5Path = cache_path + "claims10000Top5.json"

    # If already computed just return it
    if os.path.isfile(claimsTop5Path):
        print("Computations already done. Loading results from: ", claimsTop5Path)
        return openJsonDict(claimsTop5Path)

    # load wiki
    wikiArticles = parse_wiki(wiki_pages_path, wiki_parsed_cache_path)

    # Compute vocabulary size and collection frequency (using result from q1)
    wordsDictionnary = getTextStatistics()
    vocSize = len(wordsDictionnary.keys())
    collectionFrequency = sum(wordsDictionnary.values())
    avgWordPerDocument = float(collectionFrequency) / len(wikiArticles)
    print(
        "Vocabulary size: {}, collection frequency: {}, avg word per document: {}".format(vocSize, collectionFrequency,
                                                                                          avgWordPerDocument))

    claimsTop5 = {}
    for claim in tqdm(claims):
        claimWords = removeStopWords(splitWords(claim['claim']))
        top5 = []

        # Find best documents
        for docId in relevantDocs[str(claim['id'])]:
            doc = wikiArticles[docId]
            words = splitWords(doc)

            # Compute model and score for document using dirichlet and cosine sim.
            model = computeDirichletQueryLikelihoodModel(words, claimWords, wordsDictionnary, collectionFrequency, avgWordPerDocument)
            score = computeQueryScore(model, claimWords)

            # Use heap to keep the top 5 of each claim
            if len(top5) < 5:
                heappush(top5, (score, docId))
            else:
                heappushpop(top5, (score, docId))

        claimsTop5[claim['id']] = top5

    del wikiArticles
    del wordsDictionnary

    saveDictToJson(claimsTop5, claimsTop5Path)

    return claimsTop5



def question4(nbClaims = 10000):

    # First load claims
    claims = load_dataset_json(train_path)[0:nbClaims]
    print("{} claims loaded,".format(nbClaims))

    relevantDocs = findRelevantDocumentsClaims(claims)

    # Reload this claims in case program is re-runing
    claims = []
    claimsID = list(relevantDocs.keys())
    allClaims = load_dataset_json(train_path)

    for claim in allClaims:
        if str(claim['id']) in claimsID:
            claims.append(claim)
            claimsID.remove(str(claim['id']))

        if len(claimsID) == 0:
            break
    del allClaims
    print('Claims loaded: ',len(claims))

    top5Docs = computeTop5dirichlet(claims,relevantDocs)

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
