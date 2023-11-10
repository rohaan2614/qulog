"""
Utilities to be used by main code

"""
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy

nlp = spacy.load("en_core_web_sm")


def test_train_split(
    src_path, train_path, test_path, validate_path, test_size=0.33, rand_state=10
):
    # sourcery skip: extract-duplicate-method
    """
    Takes in the nine systems data provided by the QuLog Project and splits it into training and test data.
    """

    df = pd.read_csv(src_path)
    df = df.drop(
        df[
            (df.log_level != "info")
            & (df.log_level != "warn")
            & (df.log_level != "error")
        ].index
    )
    x_train, x_test, y_train, y_test = train_test_split(
        df["static_text"], df["log_level"], test_size=test_size, random_state=rand_state
    )
    x_test, x_validate, y_test, y_validate = train_test_split(
        x_test, y_test, test_size=0.33, random_state=10
    )
    df_train = pd.DataFrame(x_train, columns=["static_text"])
    df_train["log_level"] = y_train
    print(f"[UTIL_Test_Train_Split]\tdf_train: \n {df_train}")

    df_test = pd.DataFrame(x_test, columns=["static_text"])
    df_test["log_level"] = y_test
    print(f"[UTIL_Test_Train_Split]\tdf_test: \n {df_test}")

    df_validate = pd.DataFrame(x_validate, columns=["static_text"])
    df_validate["log_level"] = y_validate
    print(f"[UTIL_Test_Train_Split]\tdf_validate: \n {df_validate}")

    df_train.to_pickle(train_path)
    df_test.to_pickle(test_path)
    df_validate.to_pickle(validate_path)


def tokenize(txt):
    """
    The function "tokenize" takes a string as input, removes any non-alphanumeric characters and extra
    spaces, tokenizes the string using the spaCy library, and returns a list of tokens.

    :param txt: The `txt` parameter is a string that represents the text input that you want to tokenize
    :return: a list of tokens.
    """

    # sourcery skip: for-append-to-extend, identity-comprehension, inline-immediately-returned-variable, list-comprehension, simplify-generator
    txt = re.sub("[^a-zA-Z0-9_]", " ", txt)
    txt = re.sub(" +", " ", txt)
    doc = nlp(txt)

    tokens = []
    for token in doc:
        tokens.append(token)

    return tokens


def pos(txt):
    """
    The function `pos` takes a text as input, cleans it by removing non-alphanumeric characters and
    extra spaces, then uses natural language processing to determine the part of speech for each word in
    the text and returns a list of the part of speech tags.
    
    :param txt: The `txt` parameter is a string that represents the text you want to analyze for
    part-of-speech tagging
    :return: The function `pos(txt)` returns a list of part-of-speech tags for each token in the input
    text.
    """
    # sourcery skip: for-append-to-extend, inline-immediately-returned-variable, list-comprehension
    txt = re.sub("[^a-zA-Z0-9_]", " ", txt)
    txt = re.sub(" +", " ", txt)
    doc = nlp(txt)
    part_of_speech = []
    for token in doc:
        part_of_speech.append(token.pos_)

    return part_of_speech


def pos_num(txt):
    """
    The function `pos_num` takes a string as input, removes any non-alphanumeric characters and extra
    spaces, tokenizes the string using spaCy's natural language processing library, and returns a list
    of part-of-speech tags for each token in the string.
    
    :param txt: The parameter `txt` is a string that represents the text you want to analyze
    :return: a list of part-of-speech tags for each token in the input text.
    """
    # sourcery skip: for-append-to-extend, inline-immediately-returned-variable, list-comprehension
    txt = re.sub("[^a-zA-Z0-9_]", " ", txt)
    txt = re.sub(" +", " ", txt)
    doc = nlp(txt)
    part_of_speech = []
    for token in doc:
        part_of_speech.append(token.pos)

    return part_of_speech


def preprocess(file):
    """
    The `preprocess` function reads a pickle file into a pandas DataFrame, adds three new columns to the
    DataFrame by applying different functions to the "static_text" column, prints the first few rows of
    the DataFrame, and saves the modified DataFrame as a new pickle file.
    
    :param file: The `file` parameter is the path to the pickle file that contains the data you want to
    preprocess
    """
    df = pd.read_pickle(file)
    df["tokens"] = df["static_text"].apply(tokenize)
    df["pos"] = df["static_text"].apply(pos)
    df["pos_num"] = df["static_text"].apply(pos_num)

    print(df.head())
    print(df.iloc[0, 2])
    print(df.iloc[0, 3])
    print(df.iloc[0, 4])

    file_new = file.replace(".pickle", "_tokenized.pickle")
    df.to_pickle(file_new)
