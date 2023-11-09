import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split


#Takes in the nine_systems data provided by the QuLog Project and splits it into training and test data.


df = pd.read_csv('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_In/nine_systems_data.csv')

df = df.drop(df[(df.log_level != "info") & (df.log_level != "warn") & (df.log_level != "error")].index)



X_train, X_test, y_train, y_test = train_test_split(df['static_text'], df['log_level'], test_size=0.33, random_state=10)

X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test,  test_size=0.33, random_state=10)




df_Train = pd.DataFrame(X_train, columns = ['static_text'])

df_Train['log_level'] = y_train

print(df_Train)



df_Test = pd.DataFrame(X_test, columns = ['static_text'])

df_Test['log_level'] = y_test

print(df_Test)


df_Validate = pd.DataFrame(X_validate, columns = ['static_text'])

df_Validate['log_level'] = y_validate

print(df_Validate)




df_Train.to_pickle('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_Out/training_data.pickle')
df_Test.to_pickle('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_Out/test_data.pickle')
df_Test.to_pickle('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_Out/validate_data.pickle')