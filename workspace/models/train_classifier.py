import DataScienceHelperLibrary as dsh

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sqlalchemy import create_engine

from DisasterConfig import DBSettings
from DisasterConfig import TokenizerSettings

import sys


def tokenize(text):
    '''
    Tokenize text based on config in DasterConfig.py
    
    INPUT:
    text: string: text
    
    OUTPUT:
    list of strings
    '''
    text = text.lower() # I prefer calling this function once instead for each word
    
    if TokenizerSettings.ReplaceUrlWithPlaceHolder:
        if not dsh.IsNullOrEmpty(TokenizerSettings.CleanUrlReg):
            foundUrls = re.findall(TokenizerSettings.CleanUrlReg, text)

            for url in foundUrls:
                text = text.replace(url, TokenizerSettings.UrlPlaceHolder)
    
    if TokenizerSettings.RemovePunctuation:
        text = re.sub(TokenizerSettings.RemovePuncReg, ' ', text.lower())
    
    tokens = word_tokenize(text)

    # Call str() if values have been passed to whitelistwords that are no strings
    whiteList = [str(white).lower() for white in TokenizerSettings.WhiteListWords]
    
    Lemmatizer = WordNetLemmatizer()
    Stemmer = PorterStemmer()
    
    cleanTokens = []
    for tok in tokens:

        if TokenizerSettings.ConsiderOnlyLetters:
            tok = dsh.KeepLetters(tok)
        elif TokenizerSettings.ConsiderOnlyLettersNumbers:
            tok = dsh.KeepLettersNumbers(tok)
            
        if TokenizerSettings.UseLemmatizer:
            tok = Lemmatizer.lemmatize(tok)
        
        if TokenizerSettings.UseStemmer:
            stemmed = Stemmer.stem(tok)
        
        if len(tok) < TokenizerSettings.ConsiderMinimumLength:
            #print('Too short')
            continue
        
        isStopWord = False
        if TokenizerSettings.UseStopWords:
            try:
                for lang in list([lng for lng in TokenizerSettings.SupportedLanguages if lng is not None and len(lng) > 0]):
                    if tok in stopwords.words(lang):
                        isStopWord = True
                        break
            except:
                print('Error during stop word handling with language: ', lang, \
                    '\nand token: ', tok)
        if isStopWord:
            #print('Stopword')
            continue
        
        isUnknownWord = False
        if TokenizerSettings.UseWordNet:
            if not wordnet.synsets(tok):
                if not tok in whiteList:
                    isUnknownWord = True
        if isUnknownWord:
            continue
        
        cleanTokens.append(tok)
    
    return cleanTokens

def load_data(database_filepath):
    '''
    Load table configured in DistaterConfig.py from given sql lite database, clean data (remove a few not usable messages), and split data into X and Y (messages and categories), 
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    df = pd.read_sql_table(DBSettings.Table, engine)
    dsh.AnalyzeDataFrame(df)
    
    # As I mentioned at ETL 7.0, I didn't drop rows that were irrelevant because its possible that irrelevant rows have been added before selecting the data from database. 
# 
# So here at this step the data needs to be cleaned either way. My following steps:
# - remove column 'original' as there are many missing values as explored during ETL and cannot be proven here as you can see above ^^
# - remove rows when message starts with 'NOTES:'
# - remove rows with related not in [0, 1]
# - remove rows with messages that are much longer than the length of all messages (I intend to predict messages, no prosa texts)

    df = dsh.RemoveColumnsByWildcard(df, 'original')
    df = dsh.RemoveRowsWithValueInColumn(df, 'message', ['NOTES:'], option = 'startswith')
    df = dsh.SelectRowsWithValueInColumn(df, 'related', [0, 1])
    df = dsh.RemoveRowsByValuesOverAverage(df, 'message', 4)
    
    X = dsh.SelectColumnsByWildcard(df, ['message'])
    Y = dsh.RemoveColumnsByWildcard(df, ['index', 'id', 'message', 'original', 'genre'])
    print('Target columns: ' ,Y)
    
    return X, Y, list(Y.columns)

def build_model():
    '''
    Create a pipeline containing: CountVectorizer, TfidfTransformer, RandomForestClassifier and MultiOutputClassifier.
    
    OUTPUT:
    cv: GridsearchCV-Object 
    '''
    _cv = CountVectorizer(tokenizer = tokenize)
    _tfidf = TfidfTransformer(use_idf = True)
    _classifierInner = RandomForestClassifier(n_estimators = 20, min_samples_split = 5)
    _classifierOuter = MultiOutputClassifier(_classifierInner)

    pipeline = Pipeline([('vect', _cv),
                         ('tfidf', _tfidf),
                         ('clf', _classifierOuter)
                        ])
    
    # 
    parameters = {
        #'clf__estimator__min_samples_split' : [10, 20],
        #'clf__estimator__n_estimators' : [10, 20],
    #    'tfidf__use_idf':[True, False], # definitely use it
        #'vect__min_df': [1, 5],
        'vect__ngram_range': ((1, 1), (1, 2)),
    #    'clf__estimator__kernel': ['poly'], 
    #    'clf__estimator__degree': [1, 2, 3],
    #    'clf__estimator__C':[1, 10, 100],

    #    'clf__estimator__max_features' : [ 'auto', 'sqrt', 'log2'],
    #    'clf__estimator__max_depth' : [2, 4, 8],
    #    'clf__estimator__min_samples_split' : [10, 15, 20],
    #    'clf__estimator__min_samples_leaf' : [5, 10],
        'clf__estimator__bootstrap' : [True, False],
    #    'clf__n_jobs' : [4],
    #    'clf__estimator__n_jobs' : [4],
    #    'clf__estimator__random_state' : [42],
    }
    
    scorer = make_scorer(dsh.MultiClassifierScoreF1)
    cv = GridSearchCV(
        pipeline, 
        param_grid = parameters,

        # doc: Controls the verbosity: 
        #      the higher, the more messages.
        verbose = 10 ,

        scoring = scorer
    )
    
    return cv
    
def TestModel(model, xTest, yTest, columns):
    '''
    Test model accuracy for each category with ClassificationReport.
    
    INPUT:
    model: A model providing predict function for given data.
    xTest: np array: Test messages as 
    yTest: np array: Categories to given messages
    columns: list of column names    
    '''
    yPred = model.predict(xTest)
    
    dicClassifyReports = {}
    
    for ind in range(yPred.shape[1]):
        colName = columns[ind]
        dsh.PrintLine('Column: ' + str(colName))
        cr = classification_report(yTest[:, ind], yPred[:, ind])
        dicClassifyReports[colName] = cr
        print(cr)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Test model accuracy for each category with ClassificationReport.
    
    INPUT:
    model: A model providing predict function for given data.
    X_test: Dataframe: Test messages as 
    Y_test: Dataframe: Categories to given messages
    columns: list of column names    
    '''
    TestModel(model, X_test['message'].values, Y_test.values, category_names)


def save_model(model, model_filepath):
    '''
    Save a model to given file path.
    '''
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    '''
    Start script with given parameters: database_filepath, model_filepath.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        if DBSettings.UseConfig:
            database_filepath = DBSettings.Database
            model_filepath = DBSettings.SaveFilePickle
        
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test =  dsh.SplitDataTrainTest(X, Y, testSize = 0.2, randomState = 42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        #model.fit(X_train['message'].values, Y_train.values)
        dsh.TrainModel(model, X_train['message'].values, Y_train.values)
        
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