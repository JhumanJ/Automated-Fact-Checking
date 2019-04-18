from lib.datasets_parsing import *
from lib.utils import *

from paths import *
from tqdm import tqdm

import os, json, gc, time, random
import pandas as pd


def arrayToSentenceData(claims):
    # claims structure: dict with id, verifiable, label, claim, evidence array

    wikiArticlesLines = parse_wiki_lines(wiki_pages_path, wiki_parsed_lines_cache_path)


    dataset = []

    for claim in tqdm(claims):

        if claim['label'] in ['SUPPORTS','REFUTES']:
            claimEvidenceSentences = []

            # Find sentences
            for evidenceGroup in claim['evidence']:
                for evidence in evidenceGroup:
                    if evidence[2]!= '' and not evidence[2] is None and evidence[2] in wikiArticlesLines:
                        claimEvidenceSentences.append(wikiArticlesLines[evidence[2]][str(evidence[3])])

            # build dataset
            if len(claimEvidenceSentences)> 0:
                dataset.append([claim['label'],claim['claim']+"".join(claimEvidenceSentences)])

    return dataset

def question6():

    trainingTextDataPath = cache_path + 'trainingTextDataPath.json'

    # Training set computation
    if os.path.isfile(trainingTextDataPath):
        print("Computations already done. Loading results from: ", trainingTextDataPath)
        dict = openJsonDict(trainingTextDataPath)
        dataset = dict["data"]
    else:
        traininClaims = load_dataset_json(train_path)
        dataset = arrayToSentenceData(traininClaims)
        saveDictToJson({'data':dataset},trainingTextDataPath)

    dataset = pd.DataFrame(dataset,columns=['label','data'])
    print(dataset.head())


question6()
