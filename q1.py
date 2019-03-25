
from lib.datasets_parsing import *
from lib.utils import *
from lib.graph import *
from tqdm import tqdm
import string, math, os

from paths import *

"""
Parse all wiki articles (cache them if needed)
Then build a dictionnary of word frequency
"""
def getTextStatistics():

    outputFilename = output_path+'termFrequencies.json'

    # Check if result file already exists and return it
    if os.path.isfile(outputFilename):
        return openJsonDict(outputFilename)

    # If file does not exist, we create it
    wikiArticles = parse_wiki(wiki_pages_path,wiki_parsed_cache_path)
    print("Dictionary created. Starting to count frequencies of terms.")
    wordsDictionnary = {}

    for key, value in tqdm(wikiArticles.items()):
        # Parse text in words. Split words using regex for words of minimal length 2
        words = splitWords(value)

        for word in words:
            if word in wordsDictionnary:
                wordsDictionnary[word] += 1
            else:
                wordsDictionnary[word] = 1

    print("Dictionary created. Now saving it.")
    saveDictToJson(wordsDictionnary,outputFilename)

    return wordsDictionnary

"""
Question 1
Create dataset of frequencies, plot it to verify Zip's law
"""
def question1():
    wordsDictionnary = getTextStatistics()
    # First graph of rank vs frequency
    lineChart(sorted(list(wordsDictionnary.values()),reverse=True),
              x_label="Rank (by decreasing frequency)",
              y_label="Frequency",
              title="Word distribution")

    # Now create secund graph with log values
    sortedDictionnary = sorted(wordsDictionnary.items(), key=lambda kv: kv[1])
    totalLength = len(sortedDictionnary)

    logDatasetX = []
    logDatasetY = []

    for idx, value in enumerate(sortedDictionnary):
        logDatasetX.append(math.log(totalLength-idx))
        logDatasetY.append(math.log(value[1]))

    lineChart(logDatasetX,logDatasetY,x_label="log(rank)",y_label="log(frequency)",title="Word distribution (log)")

    # Now plot zip law: rank * frequency
    rankDatasetX = []

    for idx, value in enumerate(sortedDictionnary):
        rankDatasetX.append((totalLength - idx) * value[1])

    lineChart(rankDatasetX, x_label="rank", y_label="rank*frequency", title="Zip's Law")
    print("Avg k: {}".format(sum(rankDatasetX)/totalLength))
    print("Total length: {}".format(totalLength))
    print("So k is almost the same as the last rank.")

# question1()
wikiArticles = parse_wiki(wiki_pages_path,wiki_parsed_cache_path)
print(list(wikiArticles.keys())[2397744])