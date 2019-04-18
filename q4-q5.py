
from lib.datasets_parsing import *
from lib.stats import *
from lib.utils import *

from paths import *
from tqdm import tqdm
from heapq import heappush, heappushpop
from q1 import getTextStatistics
from gensim.models import KeyedVectors
from lib.logistic_regression import LogisticRegression

import os, json, gc, time, random

"""
Uses drichlet to compute 5 most relevant document for each claims.
To speed up computations pre-filters documents to keep only relevant ones.

To do so, we use the inverted index
"""
def findRelevantDocumentsClaims(nbClaims=10000):

    # Generate some claims
    claims = load_dataset_json(train_path)[0:nbClaims]
    print("{} claims loaded,".format(nbClaims))

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

def computeTop5dirichlet():
    claimsTop5Path = cache_path + "claims10000Top5.json"

    # If already computed just return it
    if os.path.isfile(claimsTop5Path):
        print("Computations already done. Loading results from: ", claimsTop5Path)
        return openJsonDict(claimsTop5Path)

    relevantDocs = findRelevantDocumentsClaims()

    # If relevantdocs were generated, find corresponding claims
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
    print('Claims loaded: ', len(claims))

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

            # Compute model and score for document using dirichlet
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

def loadGoogleWord2Vec(path='../GoogleNews-vectors-negative300.bin'):

    print('Loading google trained word2vec.')
    start = time.time()
    model = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
    print("Google model loaded in {} sec.".format(time.time()-start))

    return model

def buildDataset(top5Docs):

    # Load claim used in top5Docs
    claims = []
    claimsID = list(top5Docs.keys())
    allClaims = load_dataset_json(train_path)

    for claim in allClaims:
        if str(claim['id']) in claimsID:
            claims.append(claim)
            claimsID.remove(str(claim['id']))

        if len(claimsID) == 0:
            break
    del allClaims
    print('Claims loaded: ',len(claims))

    # Now we load documents lines
    wikiArticlesLines = parse_wiki_lines(wiki_pages_path, wiki_parsed_lines_cache_path)

    # Now count for how many claims top 5 match with evidences
    count = 0
    dataset = []

    for claim in claims:

        # doc ids of our top5
        docIDs = set([doc[1] for doc in top5Docs[str(claim['id'])]])

        usedDocIDs = set()

        # add claim evidence in the dataset for this claim starting with
        # the positive examples
        claimSentences = []
        for evidenceGroup in claim['evidence']:
            for evidence in evidenceGroup:
                if evidence[2] in docIDs:
                    # Then register evidence sentence tuple (1 as relevant)
                    usedDocIDs.add(evidence[2])
                    claimSentences.append( (wikiArticlesLines[evidence[2]][str(evidence[3])],1))

        if len(usedDocIDs) == 0 or len(claimSentences) == 0:
            continue

        # Now to make a balanced dataset, for each positive evidence created for this claims
        # we have to create a negative examples (taken from top5 docs in which there was evidences)
        count = len(claimSentences)
        limit = 500

        usedDocIDs = list(usedDocIDs)
        docIndex = random.randint(0,len(usedDocIDs)-1)
        sentenceIndex = random.randint(0,len(wikiArticlesLines[usedDocIDs[docIndex]])-1)
        while count > 0:

            candidateDocLines = wikiArticlesLines[usedDocIDs[docIndex]]
            candidateSentence = candidateDocLines[str(sentenceIndex)]

            # Sentence not used in positive
            if not (candidateSentence,1) in claimSentences and candidateSentence != '':
                claimSentences.append((candidateSentence,0))
                count-=1

            docIndex = random.randint(0,len(usedDocIDs)-1)
            sentenceIndex = random.randint(0,len(wikiArticlesLines[usedDocIDs[docIndex]])-1)

            limit -=1

            if limit == 0:
                print("Error while sampling evidence from claim {}. Could not find an available sentence to be a non-relevant. PAssing".format(claim['id']))
                claimSentences = None
                break
        if claimSentences is None:
            continue

        # Now add the examples
        for sentence,relevant in claimSentences:
            # compute list of word and concat claim
            words = removeStopWords(splitWords(sentence)) + removeStopWords(splitWords(claim['claim']))
            dataset.append([words,relevant])

    print ("Dataset generated containing {} sentences".format(len(dataset)))

    return dataset

def matrixifyDataset(dataset=None):

    matrixifyDatasetFile = cache_path + 'matrixifyDatasetFile.json'

    if dataset is None:
        saveResult = True
        # If already computed just return it
        if os.path.isfile(matrixifyDatasetFile):
            print("Computations already done. Loading results from: ", matrixifyDatasetFile)
            dict = openJsonDict(matrixifyDatasetFile)
            return dict["X"],dict["Y"]

        # if dataset not generated, Compute dataset of sentences
        top5Docs = computeTop5dirichlet()
        dataset = buildDataset(top5Docs)
    else:
        saveResult = False

    # Load google word2vector trained dataset
    model = loadGoogleWord2Vec()

    X = []
    Y = []

    for data, prediction in dataset:
        count = 0
        sentence = [0]*300
        for word in data:
            if word in model:
                count += 1
                # Sum all dimensions
                for index, value in enumerate(model[word]):
                    sentence[index] += value
        # Finally average everything
        for index, value in enumerate(sentence):
            sentence[index] = sentence[index] / float(count)

        X.append(sentence)
        Y.append(prediction)

    print("{} sentences generated to train/test the model.".format(len(Y)))
    if saveResult:
        print("Saving results...")
        saveDictToJson({"X":X,"Y":Y},matrixifyDatasetFile)

    return X,Y

def testDataset(balanced = True):

    testDatasetLogisticRegressionFile = cache_path + "testDatasetLogisticRegressionFile.json"

    # If already computed just return it
    if os.path.isfile(testDatasetLogisticRegressionFile) and balanced:
        print("Computations already done. Loading results from: ", testDatasetLogisticRegressionFile)
        return openJsonDict(testDatasetLogisticRegressionFile)['data']

    claimsID = [137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708]
    # Load claims
    claims = []
    allClaims = load_dataset_json(labeled_development_path)

    for claim in allClaims:
        if int(claim['id']) in claimsID:
            claims.append(claim)
            claimsID.remove(int(claim['id']))

        if len(claimsID) == 0:
            break
    del allClaims
    print('Claims loaded: ',len(claims))

    # Now collect sentences for each
    # Now we load documents lines
    wikiArticlesLines = parse_wiki_lines(wiki_pages_path, wiki_parsed_lines_cache_path)

    # Now count for how many claims top 5 match with evidences
    count = 0
    dataset = []

    for claim in claims:

        usedDocIDs = set()

        claimSentences = []
        for evidenceGroup in claim['evidence']:
            for evidence in evidenceGroup:
                # Then register evidence sentence tuple (1 as relevant)
                usedDocIDs.add(evidence[2])
                claimSentences.append( (wikiArticlesLines[evidence[2]][str(evidence[3])],1))

        # Now to make a balanced dataset, for each positive evidence created for this claims
        if balanced:
            count = len(claimSentences)
            limit = 500

            usedDocIDs = list(usedDocIDs)
            docIndex = random.randint(0,len(usedDocIDs)-1)
            sentenceIndex = random.randint(0,len(wikiArticlesLines[usedDocIDs[docIndex]])-1)
            while count > 0:

                candidateDocLines = wikiArticlesLines[usedDocIDs[docIndex]]
                candidateSentence = candidateDocLines[str(sentenceIndex)]

                # Sentence not used in positive
                if not (candidateSentence,1) in claimSentences and candidateSentence != '':
                    claimSentences.append((candidateSentence,0))
                    count-=1

                docIndex = random.randint(0,len(usedDocIDs)-1)
                sentenceIndex = random.randint(0,len(wikiArticlesLines[usedDocIDs[docIndex]])-1)

                limit -=1

                if limit == 0:
                    print("Error while sampling evidence from claim {}. Could not find an available sentence to be a non-relevant. PAssing".format(claim['id']))
                    claimSentences = None
                    break
            if claimSentences is None:
                continue
        else:
            # Create an unbalanced dataset by adding all sentences
            limit = 1000

            usedDocIDs = list(usedDocIDs)
            docIndex = 0
            sentenceIndex = 0
            while docIndex < len(usedDocIDs):

                candidateDocLines = wikiArticlesLines[usedDocIDs[docIndex]]
                try:
                    candidateSentence = candidateDocLines[str(sentenceIndex)]
                except:
                    print(candidateDocLines.keys())
                    raise Exception("error with keys")

                # Sentence not used in positive
                if not (candidateSentence,1) in claimSentences and candidateSentence != '':
                    claimSentences.append((candidateSentence,0))



                docIndex = random.randint(0,len(usedDocIDs)-1)
                sentenceIndex +=1

                if sentenceIndex >= len(candidateDocLines.keys()) or not str(sentenceIndex) in candidateDocLines.keys():
                    docIndex+=1
                    sentenceIndex = 0

                limit -=1

                if limit == 0:
                    print("Error while sampling evidence from claim {}. Could not find an available sentence to be a non-relevant. PAssing".format(claim['id']))
                    claimSentences = None
                    break
            if claimSentences is None:
                continue

        # Now add the examples
        for sentence,relevant in claimSentences:
            # compute list of word and concat claim
            words = removeStopWords(splitWords(sentence)) + removeStopWords(splitWords(claim['claim']))
            dataset.append([words,relevant])

    print ("Dataset generated containing {} sentences".format(len(dataset)))
    saveDictToJson({"data":dataset},testDatasetLogisticRegressionFile)

    return dataset

"""
Take a list of tuple and compute some metrics
"""
def relevanceEvaluation(results):

    TP = sum([1 if item == (1,1) else 0 for item in results])
    TN = sum([1 if item == (0,0) else 0 for item in results])
    FP = sum([1 if item == (1,0) else 0 for item in results])
    FN = sum([1 if item == (0,1) else 0 for item in results])

    accuracy = float(TP+TN)/(TP+TN+FP+FN)
    precision = float(TP)/(TP+FP)
    recall = float(TP)/(TP+FN)
    f1 = 2 * (recall * precision) / (recall + precision)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



def question4():

    testXYLogisticRegression = cache_path + 'testXYLogisticRegression.json'
    testXYLogisticRegressionUnbalanced = cache_path + 'testXYLogisticRegressionUnbalanced.json'

    # Query built dataset to train the regression
    X,Y = matrixifyDataset()

    print('Generate balanced test dataset.')
    # Build test dataset. If computed before just return it
    if os.path.isfile(testXYLogisticRegression):
        print("Computations already done. Loading results from: ", testXYLogisticRegression)
        dict = openJsonDict(testXYLogisticRegression)
        Xtest,Ytest = dict['X'],dict['Y']
    else:
        testSentences = testDataset()
        XtestBalanced,YtestBalanced = matrixifyDataset(testSentences)
        saveDictToJson({"X":Xtest,"Y":Ytest},testXYLogisticRegression)

    print('Generate unbalanced test dataset.')
    # Build unbalanced test dataset (using all phrases of documents as negative). If computed before just return it
    if os.path.isfile(testXYLogisticRegressionUnbalanced):
        print("Computations already done. Loading results from: ", testXYLogisticRegressionUnbalanced)
        dict = openJsonDict(testXYLogisticRegressionUnbalanced)
        XtestUnbalanced,YtestUnbalanced = dict['X'],dict['Y']
    else:
        testSentencesUnbalanced = testDataset(balanced=False)
        XtestUnbalanced,YtestUnbalanced = matrixifyDataset(testSentencesUnbalanced)
        saveDictToJson({"X":XtestUnbalanced,"Y":YtestUnbalanced},testXYLogisticRegressionUnbalanced)

    finalResults = {}
    trainingLossesEvolutions = {}

    for learningRate in [0.005,0.01,0.05,0.1,0.15,0.2]:
        for epoch in [1,10,100,1000,10000]:
            lr = LogisticRegression(learning_rate=learningRate)
            trainingLossEvolution = lr.train(X,Y,epoch)
            trainingLossesEvolutions[str(learningRate)] = trainingLossEvolution

            predictions = lr.predict(Xtest)
            comparison = list(zip(predictions,Ytest))
            relevance = relevanceEvaluation(comparison)
            finalResults[str(learningRate)+"-e"+str(epoch)+"-balanced"] = relevance
            print ("balanced: ",relevance)

            predictions = lr.predict(XtestUnbalanced)
            comparison = list(zip(predictions,YtestUnbalanced))
            relevance = relevanceEvaluation(comparison)
            finalResults[str(learningRate)+"-e"+str(epoch)+"-unbalanced"] = relevance
            print ("unbalanced: ",relevance)

    saveDictToJson(trainingLossesEvolutions,output_path+"training-loss-ev-per-learning-rate.json")
    saveDictToJson(finalResults,output_path+"q4-q5-results.json")





question4()
