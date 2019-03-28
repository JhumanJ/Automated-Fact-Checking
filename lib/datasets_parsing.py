"""

This file contains methods to parse and manipulate datasets.
Code was initially found here: https://github.com/QiangAIResearcher/Fact-Extraction-and-Verification/blob/master/fever_io.py

"""

import json
import re
from tqdm import tqdm

def parse_wiki(wikipedia_dir, doc_id_dir):
    """
    Returns a dictionary lookup from document id (URL) to document content.
    Saves the lookup in ../data/doc_id_text to speed up subsequent passes.
    """
    # doc_id_text saves the title and content of each wiki-page
    doc_id_text = dict()
    try:
        with open(doc_id_dir, "r") as f:
            print("Reading from cache ({})".format(doc_id_dir))
            for line in f:
                fields = line.rstrip("\n").split("\t")
                doc_id = fields[0]
                text = fields[1]
                doc_id_text[doc_id] = text
    except:
        print("doc_id_dir:",doc_id_dir)
        with open(doc_id_dir, "w") as w:
            print("Constructing " + str(doc_id_dir))
            for i in tqdm(range(1, 110)):  # jsonl file number from 001 to 109
                jnum = "{:03d}".format(i)
                fname = wikipedia_dir + "wiki-" + jnum + ".jsonl"
                with open(fname) as f:
                    # point=f.tell()# file pointer starting from 0
                    line = f.readline()
                    while line:
                        data = json.loads(line.rstrip("\n"))
                        doc_id = data["id"]
                        text = data["text"]
                        if text != "":
                            w.write(doc_id + "\t" + text + "\n")
                            doc_id_text[doc_id] = text
                        # point=f.tell()
                        line = f.readline()

    return doc_id_text

def parse_wiki_lines(wikipedia_dir, doc_id_dir):
    """
    Returns a dictionary lookup from document id (URL) to document lines.
    Saves the lookup to speed up subsequent passes.
    """
    # doc_id_text saves the title and content of each wiki-page
    doc_id_lines = dict()
    try:
        with open(doc_id_dir, "r") as f:
            print("Reading from cache ({})".format(doc_id_dir))
            for line in f:
                fields = line.rstrip("\n").split("\t")
                doc_id = fields[0]
                lines = json.loads(fields[1])
                doc_id_lines[doc_id] = lines
    except Exception as error:
        print("Error: {}".format(error))
        print("doc_id_dir:",doc_id_dir)
        with open(doc_id_dir, "w") as w:
            print("Constructing " + str(doc_id_dir))
            for i in tqdm(range(1, 110)):  # jsonl file number from 001 to 109
                jnum = "{:03d}".format(i)
                fname = wikipedia_dir + "wiki-" + jnum + ".jsonl"
                with open(fname) as f:
                    # point=f.tell()# file pointer starting from 0
                    line = f.readline()
                    while line:
                        data = json.loads(line.rstrip("\n"))
                        doc_id = data["id"]
                        lines = data["lines"]

                        doclines = {}
                        for l in lines.split("\n"):
                            fields = l.split("\t")
                            if fields[0].isnumeric():
                                l_id = int(fields[0])
                                l_txt = fields[1]
                                doclines[l_id] = l_txt

                        if lines != "":
                            w.write(doc_id + "\t" + json.dumps(doclines) + "\n")
                            doc_id_lines[doc_id] = doclines
                        line = f.readline()

    return doc_id_lines


def load_doclines(titles, t2jnum, filtering=True):
    """load all lines for provided titles
    Args
    titles: list of titles
    """
    if filtering:
        # select title from titles if this title is in the wiki-pages
        filtered_titles = [title for title in titles if title in t2jnum]
        print("mismatch: {} / {}".format(len(titles) - len(filtered_titles), len(titles)))
        titles = filtered_titles

    docs = {"dummy_id": [(title, "dummy_linum") for title in titles]}
    doclines = load_doc_lines(docs, t2jnum, wikipedia_dir="../data/wiki-pages/")
    return doclines


def load_doc_lines(docs=dict(), t2jnum=dict(), wikipedia_dir="../data/wiki-pages/"):
    """Returns a dictionary from titles to line numbers to line text.
    Args
    docs: {claim_id: [(title, sentence_num),  ...], ...}
    Input is a dictionary from claim ids to titles and line numbers,
    and a lookup from titles to filenumbers.
    """
    doclines = dict()
    jnums = dict()
    titles = set()
    ## cid is the claim id that is an integer
    for cid in docs:
        for title, sentence_num in docs[cid]:
            doclines[title] = dict()
            titles.add(title)
            if title in t2jnum:
                jnum, point = t2jnum[title]
                if jnum not in jnums:
                    jnums[jnum] = set()
                jnums[jnum].add(point)
            else:
                print(str(title) + " not in t2jnum!")
    for jnum in tqdm(jnums):
        points = sorted(list(jnums[jnum]))
        fname = wikipedia_dir + "wiki-" + jnum + ".jsonl"
        with open(fname) as f:
            for point in points:
                f.seek(point, 0)
                line = f.readline()
                data = json.loads(line.rstrip("\n"))
                title = data["id"]
                lines = data["lines"]
                assert title in titles
                if title in titles and lines != "":
                    for l in lines.split("\n"):
                        fields = l.split("\t")
                        if fields[0].isnumeric():
                            l_id = int(fields[0])
                            l_txt = fields[1]
                            doclines[title][l_id] = l_txt
    return doclines


def get_evidence_sentence_list(evidences, t2l2s, prependlinum=False, prependtitle=False):
    """lookup corresponding sentences and return list of sentences
    Args
    evidences: [(title, linum), ...]
    t2l2s: title2line2sentence <- output of load_doc_lines
    Returns
    list of evidence sentences
    """
    SEP = "#"

    def process_title(title):
        """ 'hoge_fuga_hoo' -> 'hoge fuga hoo' """
        return re.sub("_", " ", title)

    def maybe_prepend(title, linum):
        prep = list()
        if prependtitle:
            prep.append(title)
        if prependlinum:
            prep.append(str(linum))

        content = " {} ".format(SEP).join(prep)
        if prep:
            return "{0} {1} {0}".format(SEP, content)
        else:
            return content

    titles = [title for title, _ in evidences]
    linums = [linum for _, linum in evidences]

    return [(maybe_prepend(process_title(title), linum) + " " + t2l2s[title][linum]).strip() for title, linum in
            zip(titles, linums)]


def load_dataset_json(path, instance_num=1e6):
    """
    Reads the Fever Training set, returns list of examples.
    instance_num: how many examples to load. Useful for debugging.
    """
    data = []
    with open(path, 'r') as openfile:
        for iline, line in enumerate(openfile.readlines()):
            data.append(json.loads(line))
            if iline + 1 >= instance_num:
                break
    return data