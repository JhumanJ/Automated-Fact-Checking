
from lib.datasets_parsing import *
from lib.stats import *
from paths import *
from tqdm import tqdm

import os, json

docInvertedIndexFile = output_path + 'docInvertedIndex.json'
claimsTfIdfFile = output_path + 'claimsTfIdfFile.json'
docTfIdfFile = cache_path + 'docTfIdfFile.txt'
cosineSimFile = output_path + 'similarity-score.json'

def question2():

    wikiArticles = parse_wiki(wiki_pages_path, wiki_parsed_cache_path)
    print('Wiki articles loaded.')

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

    scores = getClaimsVsDocScore(claims, wikiArticles)
    for id in scores:
        claim = scores[id]
        top5 = []
        for docId, score in sorted(claim.items(), key=lambda kv: kv[1],reverse=True):
            if len(top5) > 5:
                break
            top5.append((docId, score))
        print("Claim id: {}\nMost similar documents:".format(id))
        for docId, score in top5:
            print("\t- Document '{}' with score {}".format(docId,score))



"""
Compute an inverted index for a set of documents
"""
def computedInvertedIndex(wikiArticles):
    # Now, build an inverted index for all documents
    # Load it, if it was already built
    if os.path.isfile(docInvertedIndexFile):
        invertedIndex = openJsonDict(docInvertedIndexFile)
        print('Inverted index loaded.')
    else:
        # Not done yet, build index
        invertedIndex = {}

        # For each article, count words and add them to the inverted index
        for key, value in tqdm(wikiArticles.items()):
            wordCounts = wordCount(splitWords(value))
            for word, value in tqdm(wordCounts.items()):
                # Init word in index if not already set
                if not word in invertedIndex:
                    invertedIndex[word] = []
                # Add word to word index (in doc)
                invertedIndex[word].append((key, wordCounts[word]))
        # Save inverted index
        saveDictToJson(invertedIndex, docInvertedIndexFile)
        print('Inverted index built.')

"""
Compute tf-idf for each word in each in the whole doc collection (for relevant docs)
"""
def computedTfIdfForDocs(claimsTfIdf,wikiArticles,invertedIndex):
    # Compute idtf for each document is a very long task, so we are going to pre-filter the set of documents
    # to keep only the one containing important words for claims.
    # We set a minimum of 0.2 as tfidf value for important words
    print("Preparing documents filtering by finding important words in claims (min tf-idf of 0.2)")
    importantWords = set()
    for result in claimsTfIdf:
        for key in result['tfidfs']:
            if result['tfidfs'][key] > 0.2:
                importantWords.add(key)

    print("{} important words found. Now filtering docs. Now finding relevant documents.".format(len(importantWords)))
    importantWords = removeStopWords(importantWords)
    relevantDocs = {}
    for id, doc in tqdm(wikiArticles.items()):
        words = splitWords(doc)
        for word in words:
            if word in importantWords:
                relevantDocs[id] = doc
                break
    print("{} relevant documents found.".format(len(relevantDocs)))

    # For each doc write a line with the tf-idf json
    print('Now computing tf-idf for documents.')
    with open(docTfIdfFile, "a") as w:
        for id, doc in tqdm(relevantDocs.items()):
            w.write(id + "\t" + json.dumps(computeTfIdf(doc, id, invertedIndex, len(wikiArticles), True)) + "\n")

    print('Doc tf-idf file computed.')

"""
Compute cosine similarity for each claim against each relevant docs
"""
def getClaimsVsDocScore(claims,wikiArticles):

    # If already computed return it
    if os.path.isfile(cosineSimFile):
        return openJsonDict(cosineSimFile)

    # Load inverted index
    invertedIndex = computedInvertedIndex(wikiArticles)

    # Compute tf-idf for each claim
    if os.path.isfile(claimsTfIdfFile):
        claimsTfIdf = openJsonDict(claimsTfIdfFile)
        print('Claims tf-idf scores loaded.')
    else:
        claimsTfIdf = computeTfIdfForClaims(claims, invertedIndex, len(wikiArticles))
        saveDictToJson(claimsTfIdf, claimsTfIdfFile)
        print('Claims tf-idf index built.')

    # Create docTfIdfFile
    if not os.path.isfile(docTfIdfFile):
        print("Cache doc id not found.")
        computedTfIdfForDocs(claimsTfIdf, wikiArticles, invertedIndex)

    # load td-idf documents
    docsTfIdf = {}
    with open(docTfIdfFile, "r") as f:
        print("Reading from cache tf-idf for docs ({})".format(docTfIdfFile))
        for line in f:
            fields = line.rstrip("\n").split("\t")
            doc_id = fields[0]
            data = json.loads(fields[1])
            docsTfIdf[doc_id] = data

    print("Docs tf-idf loaded")

    scores = {}
    for idDoc, doc in tqdm(docsTfIdf.items()):
        for claim in claimsTfIdf:
            if not claim['id'] in scores:
                scores[claim['id']] = {}

            scores[claim['id']][idDoc] = cosineSim(doc['tfidfs'], claim['tfidfs'])
    print('All similarity scores computed.')

    saveDictToJson(scores, cosineSimFile)

    return scores


question2()
