
from lib.datasets_parsing import *
from lib.stats import *
from paths import *
from tqdm import tqdm
from heapq import heappush, heappushpop

import os, json

def question4():

    # Load wiki articles
    wikiArticles = parse_wiki_lines(wiki_pages_path, wiki_parsed_lines_cache_path)

    for key,value in wikiArticles['1986_NBA_Finals'].items():
        print ("{}:{}".format(key,value))


question4()
