from lib.datasets_parsing import *
from lib.utils import *

from paths import *
from tqdm import tqdm

import os, json, gc, time, random
import pandas as pd


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

def getRNN(inputLenght,maxWords):
    inputs = Input(name='inputs',shape=[inputLenght])
    layer = Embedding(maxWords,50,input_length=inputLenght)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def question6():

    trainingTextDataPath = cache_path + 'trainingTextDataPath.json'
    testingTextDataPath = cache_path + 'testingTextDataPath.json'

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

    dataset = pd.DataFrame(dataset,columns=['label','data'])
    testDataset = pd.DataFrame(testDataset,columns=['label','data'])
    print(dataset.head())
    print(testDataset.head())

    supportCount = len(dataset[dataset['label']==1])
    print("Training: {} labels SUPPORTS and {} label REFUTES".format(supportCount, len(dataset)-supportCount))
    supportCount = len(testDataset[testDataset['label']==1])
    print("Testing: {} labels SUPPORTS and {} label REFUTES".format(supportCount, len(testDataset)-supportCount))

    # build datasets
    X_train = dataset['data']
    Y_train = dataset['label'].values
    X_test = testDataset['data']
    Y_test = testDataset['label'].values

    # Word embedding
    max_words = 1000
    max_len = 150
    tok = Tokenizer(num_words=max_words)

    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    print(X_train.head())


    # Build network
    model = getRNN(max_len,max_words)
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

    # Train network
    model.fit(sequences_matrix,Y_train,batch_size=128,epochs=100,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

    # test
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    accr = model.evaluate(test_sequences_matrix,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


question6()
