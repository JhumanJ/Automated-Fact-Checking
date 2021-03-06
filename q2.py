
from lib.datasets_parsing import *
from lib.stats import *
from lib.utils import *

from paths import *
from tqdm import tqdm

import os, json, csv

docInvertedIndexFile = output_path + 'docInvertedIndex.json'
docIdfFile = output_path + 'docIdf.json'

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

    csvData = [['claim id',	'doc id_1',	'doc id_2',	'doc id_3','doc id_4','doc id_5']]
    scores = getClaimsVsDocScore(claims, wikiArticles)
    for id in scores:
        claim = scores[id]
        top5 = []
        for docId, score in sorted(claim.items(), key=lambda kv: kv[1],reverse=True):
            if len(top5) > 5:
                break
            top5.append((docId, score))
        csvData.append([id]+[item[0] for item in top5])
        print("Claim id: {}\nMost similar documents:".format(id))
        for docId, score in top5:
            print("\t- Document '{}' with score {}".format(docId,score))

    # Save result as csv
    with open(output_path+'q2.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row in csvData:
            writer.writerow(row)

    print('Ouput CSV written.')

"""
Compute an inverted index for a set of documents
"""
def computedInvertedIndex(wikiArticles):
    # Now, build an inverted index for all documents
    # Load it, if it was already built
    if os.path.isfile(docInvertedIndexFile):
        print("Loading inverted index.")
        invertedIndex = openJsonDict(docInvertedIndexFile)
        print("Inverted index loaded. Length: {}".format(len(invertedIndex)))
    else:
        # Not done yet, build index
        invertedIndex = {}

        # For each article, count words and add them to the inverted index
        for key, value in tqdm(wikiArticles.items()):
            wordCounts = wordCount(splitWords(value))
            for word in wordCounts:
                # Init word in index if not already set
                if not word in invertedIndex:
                    invertedIndex[word] = []
                # Add word to word index (in doc)
                invertedIndex[word].append((key, wordCounts[word]))
        print("Inverted index computed. Now saving it.")
        # Save inverted index
        saveDictToJson(invertedIndex, docInvertedIndexFile)
        print('Inverted index saved.')

    return invertedIndex

def computeDocIdf(wikiArticles,documentCount):

    if os.path.isfile(docIdfFile):
        print("Loading document idf file.")
        docIdf = openJsonDict(docIdfFile)
        print("Document idf. Length: {}".format(len(docIdf)))
    else:

        # Load inverted index
        invertedIndex = computedInvertedIndex(wikiArticles)

        docIdf = {}
        for key, value in tqdm(invertedIndex.items()):
            docIdf[key] = math.log10(float(documentCount) / len(value))
        print("Document idf computed. Now saving it. Length: {}".format(len(docIdf)))
        saveDictToJson(docIdf, docIdfFile)
        print('Document idf saved.')

    return docIdf


"""
Compute tf-idf for each word in each in the whole doc collection (for relevant docs)
"""
def computedTfIdfForDocs(claimsTfIdf,wikiArticles,docIdf):


    if os.path.isfile(cache_path+'relevant-docs.json'):
        print("Loading relevant documents idf file.")
        relevantDocs = openJsonDict(cache_path+'relevant-docs.json')
        print("Relevant documents loaded. Length: {}".format(len(relevantDocs)))
    else:

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
        print("{} relevant documents found. Saving them.".format(len(relevantDocs)))
        saveDictToJson(relevantDocs,cache_path+'relevant-docs.json')
        print("Relevant docs saved.")

    # Load doc idf
    numberDocs = len(wikiArticles)

    # For each doc write a line with the tf-idf json
    print('Now computing tf-idf for documents.')
    with open(docTfIdfFile, "a") as w:
        for id, doc in tqdm(relevantDocs.items()):
            w.write(id + "\t" + json.dumps(computeTfIdf(doc, id, None, numberDocs,docIdf)) + "\n")

    print('Doc tf-idf file computed.')

"""
Compute cosine similarity for each claim against each relevant docs
"""
def getClaimsVsDocScore(claims,wikiArticles):

    # If already computed return it
    if os.path.isfile(cosineSimFile):
        return openJsonDict(cosineSimFile)

    # Load doc idf
    numberDocs = len(wikiArticles)

    # Compute tf-idf for each claim
    if os.path.isfile(claimsTfIdfFile):
        claimsTfIdf = openJsonDict(claimsTfIdfFile)
        print('Claims tf-idf scores loaded.')
    else:
        print('Building tf-idf index for claims.')
        # Load inverted index
        invertedIndex = computedInvertedIndex(wikiArticles)

        claimsTfIdf = computeTfIdfForClaims(claims, invertedIndex, numberDocs)
        saveDictToJson(claimsTfIdf, claimsTfIdfFile)
        del invertedIndex # no need for it anymore, clear memory
        print('Claims tf-idf index built.')

    # Create docTfIdfFile
    if not os.path.isfile(docTfIdfFile):
        print("Cache doc id not found.")
        docIdf = computeDocIdf(wikiArticles, numberDocs)
        computedTfIdfForDocs(claimsTfIdf, wikiArticles, docIdf)
        del docIdf # not needed anymore here

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
