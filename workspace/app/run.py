from DisasterConfig import TokenizerSettings

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

DBSettings.UseConfig = True

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

# load data
engine = create_engine('sqlite:///{}'.format(DBSettings.Database)
df = pd.read_sql_table(DBSettings.Table, engine)

# load model
model = joblib.load(DBSettings.SaveFilePickle)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result = classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()