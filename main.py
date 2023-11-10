"""
This module imports utilities from utils.py. 

Google format guide followed

Originally written by : Rob
Refactored by : Roan 

Note:
    Any additional notes or considerations can be included in the 'Note' section.

"""

# sourcery skip: dont-import-test-modules
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from utils import preprocess
from model import Model

# global constants
SRC_PATH = "Data_In/nine_systems_data.csv"
TRAIN_PATH = "df_train.pickle"
TEST_PATH = "df_test.pickle"
VALIDATE_PATH = "df_validate.pickle"


def main():
    """Main.py code"""   
    #Takes in the nine_systems data provided by the QuLog Project and splits it into training and test data.
    df = pd.read_csv(SRC_PATH)
    df = df.drop(df[(df.log_level != "info") & (df.log_level != "warn") & (df.log_level != "error")].index)

    #Pre-processing
    df = preprocess(df)
    torch.manual_seed(41)
    model = Model(10, 8, 9, 3)

    #Creates numeric values for classifications
    df['log_level_num'] = df['log_level'].replace('info', np.int64(0.0)).replace('error', np.int64(1.0)).replace('warn', np.int64(2.0))

    X = np.ndarray(shape=(len(df), model.in_features))

    #Creates numpy array for X(input) values
    print(df.iloc[0])
    print(type(df.iloc[1,6]))

    for i in range(X.shape[1]-1):
        lst = df.iloc[i,5] # using "lst" as a variable name is bad practice. We should change this Rob with something meaningful.
        for k, value in enumerate(lst):
            X[k, i] = value


    
    y = df['log_level_num']
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test,  test_size=0.33, random_state=10)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    X_validate = torch.FloatTensor(X_validate)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    y_validate = torch.FloatTensor(y_validate)


    criterion = torch.nn.CrossEntropyLoss()

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
    
    # Validate the model

    validate = model.forward(X_validate)

    print(validate)

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





if __name__ == "__main__":
    main()
