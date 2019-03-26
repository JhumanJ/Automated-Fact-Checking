
from lib.datasets_parsing import *
from lib.stats import *
from paths import *
from tqdm import tqdm
from q1 import getTextStatistics

import os, json

def question3():

    docQueryLikelihood = cache_path + 'docQueryLikelihood.txt'


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

    # Compute vocabulary size and collection frequency (using result from q1)
    wordsDictionnary = getTextStatistics()
    vocSize = len(wordsDictionnary.keys())
    collectionFrequency = sum(wordsDictionnary.values())
    print("Vocabulary size: {}, collection frequency: {}".format(vocSize,collectionFrequency))


    del wordsDictionnary

    # Now for each document, compute query likelihood model for this set of words
    # Create docQueryLikelihood
    if not os.path.isfile(docQueryLikelihood):
        print("Cached docQueryLikelihood not found.")
        with open(docQueryLikelihood, "a") as w:
            wikiArticles = parse_wiki(wiki_pages_path, wiki_parsed_cache_path)
            print('Wiki articles loaded.')


            for id, doc in tqdm(wikiArticles.items()):
                words = splitWords(doc)
                # Without smoothing
                models = {
                    'no-smooth': computeQueryLikelihoodModel(words, claimWords),
                    'laplace': computeLaplaceQueryLikelihoodModel(words, claimWords,vocSize),
                }
                w.write(id + "\t" + json.dumps(models) + "\n")
            del wikiArticles
        print('docQueryLikelihood computed.')

    # Now find top 5 for each claim
    for claim in claims:
        query = removeStopWords(splitWords(claim['claim']))
        print(claim['claim'], query)
        claimScore = {}

        with open(docQueryLikelihood, "r") as f:
            for line in tqdm(f, total=5396106):
                fields = line.rstrip("\n").split("\t")
                doc_id = fields[0]
                model = json.loads(fields[1])
                score = computeQueryScore(model, query)
                if score > 0:
                    claimScore[doc_id] = score
        print(claimScore)


question3()
