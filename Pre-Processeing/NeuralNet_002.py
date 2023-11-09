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


'''
A method to pad or truncate the numeric parts of speach tags to the correct length for the neural network
'''
def max_len(df):
    length = model.in_features

    print(df.iloc[0, 5])
    for i in range(len(df)):
        while len(df.iloc[i, 5]) < length:
            df.iloc[i, 5].append(np.int64(-1))
        df.iat[i,5] = df.iloc[i, 5][:length]

        #print(df.iloc[i, 5])
        #print('------------------')

'''
A class to represent the neural network
'''
class Model(nn.Module):

    '''
    Constructor for neural network.
    '''

    def __init__(self, in_features, h1, h2, out_features = 3):
        super().__init__()
        self.in_features = in_features
        self.h1 = h1
        self.h2 = h2


        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)


    '''
    A method to move the neural network layer forward.
    '''
    def forward(self, x):
        x = F.softmax(self.fc1(x))
        x = F.softmax(self.fc2(x))

        x = self.out(x)

        return x

'''
Creates parts of speach tags
'''
def Pos(txt):
    txt = re.sub('[^a-zA-Z0-9_]', ' ', txt)
    txt = re.sub(' +', ' ', txt)
    doc = nlp(txt)
    partOfSpeech = []
    for token in doc:
        partOfSpeech.append(token.pos_)

    return partOfSpeech

'''
Converts Tokens into numeric parts of speach tags
'''
def PosNum(txt):
    txt = re.sub('[^a-zA-Z0-9_]', ' ', txt)
    txt = re.sub(' +', ' ', txt)
    doc = nlp(txt)
    partOfSpeech = []
    for token in doc:
        partOfSpeech.append(np.int64(token.pos))

    return partOfSpeech


'''
Performs processing steps
'''
def preprocess(df):

    #Creates a column for tokenized log messages
    df['tokens'] = df['static_text'].apply(tokens)
    #Creates a column for parts of speach
    df['pos'] = df['static_text'].apply(Pos)
    #Creats a column for numeric parts of speach tags
    df['posNum'] = df['static_text'].apply(PosNum)

    return df

'''
Tokenizes the log messages
'''
def tokens(txt):

    txt = re.sub('[^a-zA-Z0-9_]', ' ', txt)
    txt = re.sub(' +', ' ', txt)
    doc = nlp(txt)

    tokens = []

    for token in doc:
        tokens.append(token)

    return tokens





#Takes in the nine_systems data provided by the QuLog Project and splits it into training and test data.



df = pd.read_csv('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_In/nine_systems_data.csv')


df = df.drop(df[(df.log_level != "info") & (df.log_level != "warn") & (df.log_level != "error")].index)


#Pre-processing

df = preprocess(df)



torch.manual_seed(41)

model = Model(10, 8, 9, 3)

#Creates numeric values for classifications
df['log_level_num'] = df['log_level'].replace('info', np.int64(0.0)).replace('error', np.int64(1.0)).replace('warn', np.int64(2.0))
#df['log_level_num'] = df['log_level_num'].replace('error', np.int64(1.0))
#df['log_level_num'] = df['log_level_num'].replace('warn', np.int64(2.0))



max_len(df)

#df_Train['posNum'] = df_Train['posNum'].astype(np.int64)

X = np.ndarray(shape=(len(df), model.in_features))

#Creates numpy array for X(input) values
print(df.iloc[0])
print(type(df.iloc[1,6]))

#print(X.shape)
#print(len(df))
#print('-----------------------------')
for i in range(X.shape[1]-1):
    list = df.iloc[i,5]
    #print(i)
    for k in range(len(list)):
        X[k, i] = list[k]


y = np.ndarray(shape=(len(df)))

#Creates numpy array for X(input) values
#print(df.iloc[0])

#print(X.shape)
#print(len(df))
#print('-----------------------------')
for i in range(len(df)):
    print(df.iloc[i,6])
    #y[i] = df.iloc[i,6]



y = df['log_level_num']
y = y.values

print(y.shape)
print(type(y[0]))
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test,  test_size=0.33, random_state=10)



X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
X_validate = torch.FloatTensor(X_validate)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
y_validate = torch.FloatTensor(y_validate)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    print(y_pred)
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

#Validate the model

validate = model.forward(X_validate)

#print(validate)

correct = 0
with torch.no_grad():
    y_eval = model.forward(X_validate) #y_eval is the prediction
    print(type(y_eval))
    #loss = criterion(y_eval, y_validate)



with torch.no_grad():
    for i, data in enumerate(X_validate):
        y_val = model.forward(data)

        print(f'{i+1}| {y_val} : {y_validate[i]}  : {y_val.argmax().item()}')

        #Correct?
        if y_val.argmax().item() == y_validate[i]:
            correct = correct+1

print(f'{correct} of {len(y_validate)}')