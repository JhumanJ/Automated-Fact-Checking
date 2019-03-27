
from lib.datasets_parsing import *
from lib.stats import *
from paths import *
from tqdm import tqdm
from q1 import getTextStatistics
from heapq import heappush, heappushpop

import os, json

def question3():

    docQueryLikelihood = cache_path + 'docQueryLikelihood.txt'
    fiveMostSimilarDoc = output_path + 'fiveMostSimilarDoc.json'

    # If already computed just return it
    if os.path.isfile(fiveMostSimilarDoc):
        print("Computation already done. Loading results from: {}",fiveMostSimilarDoc)
        print (openJsonDict(fiveMostSimilarDoc))
        return


    # Somehow claims not in order, need to find given claims
    claimsID = [75397, 150448, 214861, 156709, 129629, 33078, 6744, 226034, 40190, 76253]
    allClaims = load_dataset_json(train_path)
    claims = []

    for claim in allClaims:
        if claim['id'] in claimsID:
            claims.append(claim)
            claimsID.remove(claim['id'])

        if len(claimsID) == 0:
            break
    print('Claims loaded.')

    # Print claims:
    for claim in claims:
        print("Claim '{}': {}".format(claim['id'],claim['claim']))
    print('\n')

    # Now as we didn't save tf in the q2, we have to recompute it
    claimWords = set()
    for claim in claims:
        words = set(removeStopWords(splitWords(claim['claim'])))
        claimWords = claimWords.union(words)

    # Load wiki articles
    wikiArticles = parse_wiki(wiki_pages_path, wiki_parsed_cache_path)

    # Compute vocabulary size and collection frequency (using result from q1)
    wordsDictionnary = getTextStatistics()
    vocSize = len(wordsDictionnary.keys())
    collectionFrequency = sum(wordsDictionnary.values())
    avgWordPerDocument = float(collectionFrequency)/len(wikiArticles)
    print("Vocabulary size: {}, collection frequency: {}, avg word per document: {}".format(vocSize,collectionFrequency,avgWordPerDocument))

    # Now for each document, compute query likelihood model for this set of words
    # Create docQueryLikelihood
    if not os.path.isfile(docQueryLikelihood):
        print("Cached docQueryLikelihood not found.")
        with open(docQueryLikelihood, "a") as w:
            print('Wiki articles loaded.')


            for id, doc in tqdm(wikiArticles.items()):
                words = splitWords(doc)
                # Without smoothing
                models = {
                    'no-smooth': computeQueryLikelihoodModel(words, claimWords),
                    'laplace': computeLaplaceQueryLikelihoodModel(words, claimWords,vocSize),
                    'jelinek': computeJelinekQueryLikelihoodModel(words, claimWords, wordsDictionnary, collectionFrequency),
                    'dirichlet': computeDirichletQueryLikelihoodModel(words, claimWords, wordsDictionnary, collectionFrequency, avgWordPerDocument)
                }
                w.write(id + "\t" + json.dumps(models) + "\n")
        print('docQueryLikelihood computed.')

    # Not needed anymore
    del wikiArticles
    del wordsDictionnary

    print('Now need to compute score for each claim.')


    # Now find top 5 for each claim
    claimsScore = {}
    for claim in claims:
        # For each claim
        query = removeStopWords(splitWords(claim['claim']))
        claimScore = {
            'no-smooth': [],
            'laplace': [],
            'jelinek': [],
            'dirichlet': []
        }

        with open(docQueryLikelihood, "r") as f:
            # For each doc
            for line in tqdm(f, total=5396106):
                # For each type of smoothing
                for key in claimScore:
                    fields = line.rstrip("\n").split("\t")
                    doc_id = fields[0]
                    models = json.loads(fields[1])

                    score = computeQueryScore(models[key], query)
                    if score > 0:
                        # Use heap to keep the top 5 of each claim
                        if len(claimScore[key]) < 5:
                            heappush(claimScore[key], (score,doc_id))
                        else:
                            heappushpop(claimScore[key], (score,doc_id))
        claimsScore[claim['id']] = claimScore
    print("Best score computed for each smoothing model. Now savnig it.")
    saveDictToJson(claimsScore,fiveMostSimilarDoc)
    print("Done saving results in : {}".format(fiveMostSimilarDoc))

    print(claimsScore)
    return



question3()
