import sys
import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    Function for loading data from database. Returns dataframes
    of messages and categories.

    INPUT:
    database_filepath - path of the database

    OUTPUT:
    X - dataframe containing messages
    Y - dataframe containing categories
    Y.columns - category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('dataset', engine)
    X = df['message']
    Y = df.iloc[:, 4:]

    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenization function for processing text data.

    INPUT:
    text - the text to be tokenized

    OUTPUT:
    clean_tokens - list of cleaned and tokenized text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function sets up and builds model, returns GridSearchCV model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])

    parameters = {
    'clf__n_estimators' : [50, 100],
    'clf__criterion' : ['gini', 'entropy'],
    'clf__max_depth' : [5, 10, 20],
    'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function prints report evaluating the model.
    """
    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred, target_names=category_names))

    return


def save_model(model, model_filepath):
    """
    Function for saving the model.

    INPUT:
    model - model to be saved
    model_filpath - path to save file to
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
