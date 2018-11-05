# Disaster Response Project - Overview
For my current training, I created a text classification model and integrated it into a web application.
The model is trained on data provided by Figure8. The data consists of two files: messages.csv and categories.csv. Most of the messages are given in english, those who are/could not be translated, are not considered for model training. Each message is linked with a unique id to one or more of 36 different categories (for example aid_related, search_and_rescue, security, military, water, food, shelter). 
The test results show mostly an average precision between 77 and 98%. "Mostly" because the data is inbalanced. That means that for some categories there are not enough messages categorized as "offer", "tools", "hospitals", "shops", "aid_centers". Test data consisted of 7711 messages and the number of messages belonging to the mentioned categories is between 28 and 100.

# Installation
You need following software and packages to train and use the model:
- Anaconda Navigator 1.8.7
- Python 3.6
- Packages: flask, numpy, pandas, matplotlib, nltk, scikitlearn, seaborn, sqlalchemy


# Process & Files
The process is divided into following three parts:

1. Data processing
An ETL (extract, transform load) pipeline loads both files and joins on common id, creates new columns for each category, converts them to numeric, drops duplicates and saves data to database based on config.
This happens in process_data.py

2. Training process
A machine learning pipeline tokenizes text based on config in DasterConfig.py, Loads table configured in DistaterConfig.py from given sql lite database, cleans data (removes a few not usable messages), and splits data into X and Y (messages and categories), creates a pipeline containing: CountVectorizer, TfidfTransformer, RandomForestClassifier and MultiOutputClassifier, tests model accuracy for each category with ClassificationReport and saves a model to given file path
This happens in train_classifier.py

3. Use model
An web application loads the trained model and provides a textbox to type in a message and categorize it.
This happens in run.py

# How to use this application:
1. Decide if you want to pass file path parameters directly or use config in DisasterConfig. The parameter "UseConfig" is for that. If true, empty strings can be passed

2. Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/, enter message and categorize it.



