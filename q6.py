from lib.datasets_parsing import *
from lib.utils import *

from paths import *
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle


import os, json, gc, time, random
import pandas as pd
import numpy as np

from q4_q5 import loadGoogleWord2Vec, relevanceEvaluation

# Import for lstm text classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


embeddedTrainingTextDataPath = cache_path + 'embeddedTrainingTextDataPath.json'
embeddedTestingTextDataPath = cache_path + 'embeddedTestingTextDataPath.json'
trainingTextDataPath = cache_path + 'trainingTextDataPath.json'
testingTextDataPath = cache_path + 'testingTextDataPath.json'


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
                dataset.append([1 if claim['label']=='SUPPORTS' else 0,claim['claim']+"".join(claimEvidenceSentences)])

    return dataset

def claimEmbedding(claim,model):

    """
        Again use wordembedding of google to transform a claim into a matrix
    """

    claim = removeStopWords(splitWords(claim))
    count = 0
    sentence = [0]*300
    for word in claim:
        if word in model:
            count += 1
            # Sum all dimensions
            for index, value in enumerate(model[word]):
                sentence[index] += value
    # Finally average everything
    for index, value in enumerate(sentence):
        sentence[index] = sentence[index] / float(count)

    return sentence

def getLSTM(inputLenght,batch_size):
    inputs = Input(name='inputs',shape=[inputLenght])
    layer = Embedding(batch_size,50,input_length=inputLenght)(inputs)
    layer = LSTM(64,name='FactCheckingLstm')(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def getEmbeddedDatasets():

    # First check if embedded dataset are already prepared =======
    dataset = None
    testDataset = None

    # Training set computation
    if os.path.isfile(embeddedTrainingTextDataPath):
        print("Computations already done. Loading results from: ", embeddedTrainingTextDataPath)
        dataset = openJsonDict(embeddedTrainingTextDataPath)


    # Testing set computation
    if os.path.isfile(embeddedTestingTextDataPath):
        print("Computations already done. Loading results from: ", embeddedTestingTextDataPath)
        testDataset = openJsonDict(embeddedTestingTextDataPath)

    if not dataset is None and not testDataset is None:
        return dataset, testDataset

    # If embedded dataset are not done yet ============

    # Training set computation
    if os.path.isfile(trainingTextDataPath):
        print("Computations already done. Loading results from: ", trainingTextDataPath)
        dict = openJsonDict(trainingTextDataPath)
        dataset = dict["data"]
    else:
        claims = load_dataset_json(train_path)
        dataset = arrayToSentenceData(claims)
        saveDictToJson({'data':dataset},trainingTextDataPath)

    # Testing set computation
    if os.path.isfile(testingTextDataPath):
        print("Computations already done. Loading results from: ", testingTextDataPath)
        dict = openJsonDict(testingTextDataPath)
        testDataset = dict["data"]
    else:
        claims = load_dataset_json(labeled_development_path)
        testDataset = arrayToSentenceData(claims)
        saveDictToJson({'data':testDataset},testingTextDataPath)

    # Load google word2vector trained dataset
    model = loadGoogleWord2Vec()
    # Apply word embedding to training data using google trained word2vec
    temp = {
    'data': [],
    'label': []
    }
    for claim in tqdm(dataset):
        temp['data'].append(claimEmbedding(claim[1],model))
        temp['label'].append(claim[0])
    dataset = temp

    # Apply word embedding to test data using google trained word2vec
    temp = {
    'data': [],
    'label': []
    }
    for claim in tqdm(testDataset):
        temp['data'].append(claimEmbedding(claim[1],model))
        temp['label'].append(claim[0])
    testDataset = temp

    saveDictToJson(dataset,embeddedTrainingTextDataPath)
    saveDictToJson(testDataset,embeddedTestingTextDataPath)

    return dataset, testDataset


def question6(use_word2vec=False):

    if use_word2vec:
        dataset, testDataset = getEmbeddedDatasets()

        # build datasets
        X_train = np.array(dataset['data'])
        Y_train = np.array(dataset['label'])
        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

        X_test = np.array(testDataset['data'])
        Y_test = np.array(testDataset['label'])

        supportCount = sum(Y_train)
        print("Training ({}): {} labels SUPPORTS and {} label REFUTES".format(len(X_train),supportCount, len(X_train)-supportCount))
        supportCount = sum(Y_test)
        print("Testing ({}): {} labels SUPPORTS and {} label REFUTES".format(len(X_test), supportCount, len(X_test)-supportCount))

        # Nomrmalize the dataset
        for index in range(len(X_train)):
            X_train[index] = minmax_scale(X_train[index])
        for index in range(len(X_test)):
            X_test[index] = minmax_scale(X_test[index])

        max_len = 300
        training_zise = 1000

    else:

        # Use keras tokenizer which better represents the sentences

        # Training set computation
        if os.path.isfile(trainingTextDataPath):
            print("Computations already done. Loading results from: ", trainingTextDataPath)
            dict = openJsonDict(trainingTextDataPath)
            dataset = dict["data"]
        else:
            claims = load_dataset_json(train_path)
            dataset = arrayToSentenceData(claims)
            saveDictToJson({'data':dataset},trainingTextDataPath)

        # Testing set computation
        if os.path.isfile(testingTextDataPath):
            print("Computations already done. Loading results from: ", testingTextDataPath)
            dict = openJsonDict(testingTextDataPath)
            testDataset = dict["data"]
        else:
            claims = load_dataset_json(labeled_development_path)
            testDataset = arrayToSentenceData(claims)
            saveDictToJson({'data':testDataset},testingTextDataPath)

        max_len = 150
        training_zise = 1000

        dataset = pd.DataFrame(dataset,columns=['label','data'])
        testDataset = pd.DataFrame(testDataset,columns=['label','data'])

        # build datasets
        X_train = dataset['data']
        Y_train = dataset['label'].values
        X_test = testDataset['data']
        Y_test = testDataset['label'].values

        # Tokenize the sentences
        tok = Tokenizer(num_words=training_zise)

        tok.fit_on_texts(X_train)
        sequences = tok.texts_to_sequences(X_train)
        X_train = sequence.pad_sequences(sequences,maxlen=max_len)
        test_sequences = tok.texts_to_sequences(X_test)
        X_test = sequence.pad_sequences(test_sequences,maxlen=max_len)

        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

    # Build network
    model = getLSTM(max_len,training_zise)
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

    # Train network
    # todo: increase number of epochs
    print("Starting to train")
    model.fit(X_train,Y_train,batch_size=128,epochs=5,verbose=1,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

    # testing model
    results = model.predict(X_test)

    actual_results = [0 if result <0.5 else 1 for result in results]
    comparison = list(zip(actual_results,Y_test))
    print(relevanceEvaluation(comparison))

    # Genreate output File
    testClaims = load_dataset_json(labeled_development_path)
    filteredTestClaims = []

    wikiArticlesLines = parse_wiki_lines(wiki_pages_path, wiki_parsed_lines_cache_path)
    for claim in tqdm(testClaims):

        if claim['label'] in ['SUPPORTS','REFUTES']:
            claimEvidenceSentences = []

            # Find sentences
            for evidenceGroup in claim['evidence']:
                for evidence in evidenceGroup:
                    if evidence[2]!= '' and not evidence[2] is None and evidence[2] in wikiArticlesLines:
                        claimEvidenceSentences.append(wikiArticlesLines[evidence[2]][str(evidence[3])])

            # build dataset
            if len(claimEvidenceSentences)> 0:
                filteredTestClaims.append(claim)



    with open(output_path+'predictions.jsonl', 'w') as outfile:

        for idx, claim in tqdm(enumerate(filteredTestClaims)):
            if claim['label'] in ['SUPPORTS','REFUTES']:

                jsonDict = {
                    "id": claim['id'],
                    "predicted_label": ('SUPPORTS' if actual_results[idx] == 1 else 'REFUTES'),
                    "predicted_evidence": []
                }

                json.dump(jsonDict, outfile)
                outfile.write('\n')


    # Test set
    # {'accuracy': 0.6246881850480006, 'precision': 0.5747553461399058, 'recall': 0.9588813303099017, 'f1': 0.718712820803354}



question6()
