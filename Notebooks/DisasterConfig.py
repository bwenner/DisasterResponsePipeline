class DBSettings:
    Database = 'MyDisasterResponse.db'
    Table = 'Message'
    SaveFileJoblib = 'MyDisasterPredictor.pkl'
    SaveFilePickle = 'MyDisasterPredictor.sav'
	
	
class TokenizerSettings:
    
    SupportedLanguages = [
    'english', 
    #'frensh',
    #'haitian'
    ]
    
    ConsiderOnlyLetters = 0
    ConsiderOnlyLettersNumbers = 0
    ConsiderMinimumLength = 0
    
    ReplaceUrlWithPlaceHolder = False
    CleanUrlReg = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    UrlPlaceHolder = ''    
    
    RemovePunctuation = True
    RemovePuncReg = r'[^a-zA-Z0-9]'
    
    UseLemmatizer = 0
    UseStemmer = 1
    UseStopWords = 1
    UseTokenizer = 1
    UseWordNet = 1
    
    
    # If using the option UseWordNet:
    
    # By hazard I saw a message with that organization/institution,
    # filtered roughtly 40 rows and decided to consider/give this word a weight
    # and not just drop it.
    # So I intended to make it parameterizable.
    WhiteListWords = set(['UNHCR'])