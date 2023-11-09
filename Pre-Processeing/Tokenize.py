import re

import pandas as pd

from spacy.tokens import Doc

import spacy

nlp = spacy.load("en_core_web_sm")


def tokens(txt):
    txt = re.sub('[^a-zA-Z0-9_]', ' ', txt)
    txt = re.sub(' +', ' ', txt)
    doc = nlp(txt)

    tokens = []

    for token in doc:
        tokens.append(token)

    return tokens

def Pos(txt):
    txt = re.sub('[^a-zA-Z0-9_]', ' ', txt)
    txt = re.sub(' +', ' ', txt)
    doc = nlp(txt)
    partOfSpeech = []
    for token in doc:
        partOfSpeech.append(token.pos_)

    return partOfSpeech

def PosNum(txt):
    txt = re.sub('[^a-zA-Z0-9_]', ' ', txt)
    txt = re.sub(' +', ' ', txt)
    doc = nlp(txt)
    partOfSpeech = []
    for token in doc:
        partOfSpeech.append(token.pos)

    return partOfSpeech

def preprocess(file):
    df = pd.read_pickle(file)


    df['tokens'] = df['static_text'].apply(tokens)
    df['pos'] = df['static_text'].apply(Pos)
    df['posNum'] = df['static_text'].apply(PosNum)

    print(df.head())
    print(df.iloc[0, 2])
    print(df.iloc[0,3])
    print(df.iloc[0, 4])

    fileNew = file.replace(".pickle", "_tokenized.pickle")
    df.to_pickle(fileNew)


preprocess('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_Out/test_data.pickle')
preprocess('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_Out/training_data.pickle')
preprocess('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_Out/validate_data.pickle')