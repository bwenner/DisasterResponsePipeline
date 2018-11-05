
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import DataScienceHelperLibrary as dsh
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import numpy as np
import pandas as pd
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
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sqlalchemy import create_engine


# In[2]:


# load data from database
database = 'MyDisasterResponse.db'
table = 'Message'

engine = create_engine('sqlite:///{}'.format(database))

df = pd.read_sql_table(table, engine)


# In[3]:


dsh.AnalyzeDataFrame(df)


# Hmm... there were nan values that have been inserted into database...

# In[4]:


df[df.original == np.nan]


# Hmmm.... there were definitely missing values...

# In[5]:


type(df.at[7408, 'original']) # example ID I radonmly found.


# In[6]:


df[df.original == None]


# In[7]:


type(None)


# Hmmm.... I thought that None or np.nan would be interpretet as something like DBNULL. Instead it is a string value?! I didn't calculated that...

# In[8]:


df[df.original == 'NoneType']


# I give up... I thought there is an adequate handling for np.nan und database NULL values. I just intended to check the number of mising values before and after writing into database. But as you can see, currently I'm out of ideas how to do it ... And I don't intend to waste time with that.

# ## Remark: Cleaning again
# Before I split data, I remove those rows that would be null/none/empty after going through Tokenize() to avoid bad side effects. For example I don't konw how a training message is treated when it has no weights and is assigned to some categories.

# As I mentioned at ETL 7.0, I didn't drop rows that were irrelevant because its possible that irrelevant rows have been added before selecting the data from database. 
# 
# So here at this step the data needs to be cleaned either way. My following steps:
# - remove column 'original' as there are many missing values as explored during ETL and cannot be proven here as you can see above ^^
# - remove rows when message starts with 'NOTES:'
# - remove rows with related not in [0, 1]
# - remove rows that have no category at all
# - remove rows with messages that are much longer than the length of all messages (I intend to predict messages, no prosa texts)
# - remove rows that would be empty after message has gone through Tokenize( ) so please allow me at this place to slightly change the order of to do's.

# In[9]:


df = dsh.RemoveColumnsByWildcard(df, 'original')


# In[10]:


df[df['message'].str.startswith('NOTES:')]


# In[11]:


# Remove messages starting with 'NOTES:'

df = dsh.RemoveRowsWithValueInColumn(df, 'message', ['NOTES:'], option = 'startswith')


# In[12]:


# Just keep messages with related = 0 or 1 (drop 2 or potentially other values)

df = dsh.SelectRowsWithValueInColumn(df, 'related', [0, 1])


# In[13]:


# # Remove rows that have no category at all

# # Get all categories
# colsToSum = list(dsh.RemoveColumnsByWildcard(df, ['index', 'id', 'message', 'original', 'genre']))

# shapeBefore = df.shape
# dfRemovedNoCategory = df[df[colsToSum].sum(axis = 1) == 0]
# df = df[df[colsToSum].sum(axis = 1) > 0]

# print('Rows removed that had no category: ', shapeBefore[0] - df.shape[0])


# In[14]:


# Remove messages that are much longer than the average

df = dsh.RemoveRowsByValuesOverAverage(df, 'message', 4)


# ### 2. Write a tokenization function to process your text data

# In[15]:


class TokenizerSettings:
    
    SupportedLanguages = [
    'english', 
    #'frensh',
    #'haitian'
    ]
    
    ConsiderOnlyAlphabet = 1
    ConsiderMinimumLength = 3
    
    CleanUrlReg = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    UrlPlaceHolder = ''    
    
    UseLemmatizer = 1
    UseStemmer = 1
    UseStopWords = 1
    UseTokenizer = 1
    UseWordNet = 1

    WordMinLengthForStemmer = 5
    
    Lemmatizer = WordNetLemmatizer()
    Stemmer = PorterStemmer()
    
    # By hazard I saw a message with that organization/institution,
    # filtered roughtly 40 rows and decided to consider/give this word a weight
    # and not just drop it.
    # In my eyes this is not overfitting because new words need to be recognized that potentially have
    # influence. And I intended to make it parameterizable.
    WhiteListWords = set(['UNHCR'])
    

    
def Tokenize(text):
    
    text = text.lower() # I prefer calling this function once instead for each word
    
    if not dsh.IsNullOrEmpty(TokenizerSettings.CleanUrlReg):
        foundUrls = re.findall(TokenizerSettings.CleanUrlReg, text)
        
        for url in foundUrls:
            text = text.replace(url, TokenizerSettings.UrlPlaceHolder)
    
    tokens = word_tokenize(text)

    # Call str() if values have been passed to whitelistwords that are no strings
    whiteList = [str(white).lower() for white in TokenizerSettings.WhiteListWords]
    cleanTokens = []
    for tok in tokens:

        if TokenizerSettings.ConsiderOnlyAlphabet:
            tok = dsh.RemoveNonLetters(tok)
            if dsh.IsNullOrEmpty(tok):
          #      print('Empty')
                continue
        
        if TokenizerSettings.UseLemmatizer:
            tok = TokenizerSettings.Lemmatizer.lemmatize(tok).strip()
        
        if TokenizerSettings.UseStemmer and len(tok) >= TokenizerSettings.WordMinLengthForStemmer:
            stemmed = TokenizerSettings.Stemmer.stem(tok)
        
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
                print('Error during stop word handling with language: ', lang,                     '\nand token: ', tok)
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
        
        #if tok in cleanTokens:
            #print('Contained')
        #    continue
        
        cleanTokens.append(tok)
    
    return cleanTokens


# In[16]:


# Now remove those rows with empty message after going through Tokenize()

dsh.PrintLine('Removing potentially empty rows after Tokenize')
print('Please be paitent, this step may take some seconds')
df['dropRow'] = df['message'].apply(lambda x: 1 if len(Tokenize(x)) == 0 else 0)
dsh.PrintLine('Finished')

df[df['dropRow'] == 1]


# So these 37 "rubbish" messages can smoothly be dropped.

# In[17]:


df = dsh.RemoveRowsWithValueInColumn(df, 'dropRow', 1)

df = df.drop('dropRow', axis = 1)


# In[18]:


list(df[df.index == 22611].message)


# In[19]:


df[df.message.str.contains('UNHCR')]


# In[20]:


X = dsh.SelectColumnsByWildcard(df, ['message'])

Y = dsh.RemoveColumnsByWildcard(df, ['index', 'id', 'message', 'original', 'genre'])


# ### 3. Build a machine learning pipeline
# - You'll find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[21]:


# pipeline = Pipeline([
#         ('features', FeatureUnion([

#             ('text_pipeline', Pipeline([
#                 ('vect', CountVectorizer(tokenizer = Tokenize)),
#                 ('tfidf', TfidfTransformer())
#             ])),

#             #('starting_verb', StartingVerbExtractor())
#         ])),

#         ('clf', MultiOutputClassifier())
#     ])


_cv = CountVectorizer(tokenizer = Tokenize)
_tfidf = TfidfTransformer()
_classifierInner = RandomForestClassifier(n_estimators = 50, random_state = 42)
_classifierOuter = MultiOutputClassifier(_classifierInner)


pipeline = Pipeline([('vect', _cv),
                     ('tfidf', _tfidf),
                     ('clf', _classifierOuter,)
                    ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[22]:


XTrain, XTest, yTrain, yTest = dsh.SplitDataTrainTest(X, Y)


# In[23]:


xFitTrans = dsh.TrainModel(pipeline, XTrain['message'].values, yTrain.values)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[24]:


def TestModel(model, xTest, yTest, columns, printIncorrect = False):
    
    yPred = model.predict(xTest)
    
    dicClassifyReports = {}
    
    for ind in range(yPred.shape[1]):
        colName = columns[ind]
        dsh.PrintLine('Column: ' + str(colName))
        cr = classification_report(yTest[:, ind], yPred[:, ind])
        dicClassifyReports[colName] = cr
        print(cr)
        
        cntErr = 0
        dirError = {}
        lstFalses = []
        for cind in range(yTest.shape[1]):

            cntErr = 0
            colname = columns[cind]

            for ind in range(yTest.shape[0]):

                if yTest[ind, cind] != yPred[ind, cind]:
                    cntErr += 1
                    lstFalses.append(str(colname) + ' - ' + str(XTest.iloc[ind]['message']))

            err = cntErr * 100 / yTest.shape[0]
            dirError[str(colname)] = err

        dfError = pd.DataFrame({'Errors' : list(dirError.values())}, index = columns)
    if printIncorrect:
        for val in lstFalses:
            print(Tokenize(val))
        print('False predicted')

    dfError.sort_values('Errors',ascending = False)
        
    dsh.PrintLine()
    
    return dfError


# In[25]:


TestModel(xFitTrans, XTest['message'].values, yTest.values, yTest.columns, True)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[26]:




parameters = {
       # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
       # 'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
       # 'features__text_pipeline__vect__max_features': (None, 5000, 10000),
       # 'features__text_pipeline__tfidf__use_idf': (True, False),
      #  'clf__estimators': [RandomForestClassifier(10)],
      #  'clf__min_samples_split': [2, 3, 4],
      #  'features__transformer_weights': (
      #      {'text_pipeline': 1, 'starting_verb': 0.5},
      #      {'text_pipeline': 0.5, 'starting_verb': 1},
      #      {'text_pipeline': 0.8, 'starting_verb': 1},
      #  )
    
      #"classifier__max_depth": [3, None],
      #        "classifier__max_features": [1, 3, 10],
      #        "classifier__min_samples_split": [1, 3, 10],
      #        "classifier__min_samples_leaf": [1, 3, 10],
      #        # "bootstrap": [True, False],
      #        "classifier__criterion": ["gini", "entropy"]

    #'clf__estimator': [ AdaBoostClassifier(), RandomForestClassifier() ],
    'clf__estimator__n_estimators' : [20, 30, 50, 100],
    'clf__estimator__max_features' : [ 'auto', 'sqrt', 'log2'],
    #'clf__estimator__max_depth' : [],
    #'clf__estimator__min_samples_split' : [],
    #'clf__estimator__min_samples_leaf' : [],
    'clf__estimator__bootstrap' : [True, False],
    #'clf__n_jobs' : [4],
    #'clf__estimator__n_jobs' : [4],
    'clf__estimator__random_state' : [42],
    }
#parameters = _classifierOuter.get_params()

cv = GridSearchCV(pipeline, param_grid = parameters)


# In[ ]:


gridFit = cv.fit(XTrain['message'].values, yTrain.values)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


bestEstimator = gridFit.best_estimator_

TestModel(bestEstimator, XTest['message'].values, yTest.values, yPred, yTest.columns)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# ### 9. Export your model as a pickle file

# In[ ]:


#https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
#If you want to dump your object into one file - use:  

joblib.dump(bestEstimator, 'filename.pkl', compress = 1)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.
