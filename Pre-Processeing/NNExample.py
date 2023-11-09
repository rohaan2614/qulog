import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


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


torch.manual_seed(41)

model = Model(4, 8, 9, 3)

df = pd.read_csv('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_In/Iris.csv')

print(df.head)

df['species'] = df['species'].replace('Iris-setosa', 0.0)
df['species'] = df['species'].replace('Iris-versicolor', 1.0)
df['species'] = df['species'].replace('Iris-virginica', 2.0)

print(df.head())

X= df.drop('species', axis=1)

y = df['species']

X = X.values
y = y.values

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

print(X_train)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


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

