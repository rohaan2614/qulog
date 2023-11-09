import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import re


from spacy.tokens import Doc

import spacy
nlp = spacy.load("en_core_web_sm")

def max_len(df):
    length = model.in_features
    print(length)
    print(df.iloc[0, 4])
    for i in range(len(df)):
        while len(df.iloc[i, 4]) < length:
            df.iloc[i, 4].append(np.int64(-1))
        df.iat[i,4] = df.iloc[i, 4][:length]

        print(df.iloc[i, 4])
        print('------------------')

class Model(nn.Module):

    def __init__(self, in_features, h1, h2, out_features = 3):
        super().__init__()
        self.in_features = in_features
        self.h1 = h1
        self.h2 = h2


        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x




#Takes in the nine_systems data provided by the QuLog Project and splits it into training and test data.

#Create Train, Test and Validation Partitions

df = pd.read_csv('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_In/nine_systems_data.csv')


df = df.drop(df[(df.log_level != "info") & (df.log_level != "warn") & (df.log_level != "error")].index)

X_train, X_test, y_train, y_test = train_test_split(df['static_text'], df['log_level'], test_size=0.33, random_state=10)

X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test,  test_size=0.33, random_state=10)


df_Train = pd.DataFrame(X_train, columns = ['static_text'])

df_Train['log_level'] = y_train


df_Test = pd.DataFrame(X_test, columns = ['static_text'])

df_Test['log_level'] = y_test


df_Validate = pd.DataFrame(X_validate, columns = ['static_text'])

df_Validate['log_level'] = y_validate



#Pre-processing

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
        partOfSpeech.append(np.int64(token.pos))

    return partOfSpeech



def preprocess(df):

    df['tokens'] = df['static_text'].apply(tokens)
    df['pos'] = df['static_text'].apply(Pos)
    df['posNum'] = df['static_text'].apply(PosNum)

    print(df.head())
    print(df.iloc[0, 2])
    print(df.iloc[0,3])
    print(df.iloc[0, 4])

    return df



df_Train = preprocess(df_Train)
df_Test = preprocess(df_Test)
df_Validate = preprocess(df_Validate)


torch.manual_seed(41)

model = Model(4, 8, 9, 3)


df_Train['log_level_num'] = df_Train['log_level'].replace('log_level', 0.0)
df_Train['log_level_num'] = df_Train['log_level'].replace('log_level', 1.0)
df_Train['log_level_num'] = df_Train['log_level'].replace('log_level', 2.0)

df_Test['log_level_num'] = df_Test['log_level'].replace('log_level', 0.0)
df_Test['log_level_num'] = df_Test['log_level'].replace('log_level', 1.0)
df_Test['log_level_num'] = df_Test['log_level'].replace('log_level', 2.0)

print(df_Train.head())

print(df_Train.columns)
max_len(df_Train)

#df_Train['posNum'] = df_Train['posNum'].astype(np.int64)

X_train = np.ndarray(shape=(model.in_features, len(df_Train)))

for i in range(len(X_train)):
    list = df_Train.iloc[i,4]
    for k in range(len(list)):
        X_train[i,k] = list[k]

print(X_train)

'''
#X_train = np.ndarray(df_Train['posNum'])

print(type(X_train[0][0]))


X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(df_Test['posNum'])

y_train = torch.LongTensor(df_Train['log_level'])
y_test = torch.LongTensor(df_Test['log_level'])


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())

    if(i % 10 == 0):
        print(f'Epoch {i}, prediction {y_pred} and loss {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.plot(range(epochs), losses)
plt.ylabel("loss")
plt.ylabel("epoch")
plt.show()
'''

