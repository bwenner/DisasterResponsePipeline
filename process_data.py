import sys
import DataScienceHelperLibrary as dsh
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    df = None
    
     succ, files = dsh.ReadCsvFiles([messages_filepath, categories_filepath])
    if not succ:
        raise ValueError('Files could not be read')    
    return df
    
    messages = files[messages_filepath]
    categories = files[categories_filepath]
    
    dsh.AnalyseDataframe(messages)
    dsh.AnalyseDataframe(categories)
    dsh.AnalyseEqualColumns(messages, categories)
    
    return dsh.QuickMerge(messages, categories)
        
def clean_data(df):
    
    ser = categories['categories']
    categories = ser.str.split(';', expand = True)
    
    # by using range(10) I hopefully ensure to get unique values
    # (please have a look at the brief discussion in workbook
    # regarding the value 'related-2' in the column related)
    uniqueCategories = dsh.GetUniqueValuesListFromColumn(
    df, 
    'categories', 
    clean = {'' : ['-' + str(i) for i in range(10)] },
    splitby = ';',
)
    categories.columns = category_colnames
    
    typelist = []
for column in uniqueCategories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: int(str(x).replace(column + '-', "")))
    # convert column from string to numeric
    
    # This step does not need to be done beuase I'm already calling int(..) in apply
    typ = str(categories[column].dtype)
    if typ in typelist:
        continue
    typelist.append(typ)
    # data is at this point already numeric
    
    print('DType/s of new columns is/are: ', typelist)
    
    df = df.drop('categories', axis = 1)
    
    dfFinal = dsh.QuickMerge(df, categories)
    
    pass


def save_data(df, database_filename):
    pass  

    engine = create_engine('sqlite:///{}'.format(database_filename))
    dsh.PrintLine('Current Tables in DB:')
    print(engine.table_names())
    dsh.PrintLine()
    
    df.to_sql(origMessages, engine, if_exists='append', chunksize=4)
    
    engine.dispose()
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()