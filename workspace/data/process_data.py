import DataScienceHelperLibrary as dsh
from DisasterConfig import DBSettings
import pandas as pd
import numpy as np
import sys

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load both files and join on common id.
    
    INPUT:
    messages_filepath: string. Path to message file
    categories_filepath: string. Path to category file
    '''
    _, files = dsh.ReadCsvFiles([
        messages_filepath, 
        categories_filepath
    ])
    
    messages = files[messages_filepath]
    dsh.AnalyzeDataFrame(messages)
    
    categories = files[categories_filepath]
    dsh.AnalyzeDataFrame(categories)
    
    dsh.AnalyzeEqualColumns(messages, categories)
    
    df = dsh.QuickMerge(messages, categories)
    
    return df


def clean_data(df):
    '''
    Create new columns for each category, make columns numeric and drop duplicates.
    
    INPUT:
    df: Dataframe
    '''
    uniqueCategories = dsh.GetUniqueValuesListFromColumn(
        df, 
        'categories', 
        clean = {'' : ['-1', '-0'] },
        splitby = ';',
    )
    
    _ = dsh.CheckIfValuesContainedInEachOther(uniqueCategories)
    
    if 'related-2' in uniqueCategories:
        uniqueCategories.remove('related-2')
    
    ser = df['categories']
    categories = ser.str.split(';', expand = True)
    categories.columns = uniqueCategories
    
    typelist = []
    for column in uniqueCategories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: str(x).replace(column + '-', ""))
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])    
        # This step does not need to be done beuase I'm already calling int(..) in apply
        typ = str(categories[column].dtype)
        if typ in typelist:
            continue
        typelist.append(typ)
    print('DType/s of new columns is/are: ', typelist)
    
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1) 
    
    df = dsh.RemoveDuplicateRows(df)
    dsh.AnalyzeNanColumns(df)
    
    return df

def save_data(df, database_filename):
    '''
    Save data to datbase.
    
    INPUT:
    df: Dataframe
    database_filename: string.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    dsh.PrintLine('Current Tables in DB:')
    print(engine.table_names())
    dsh.PrintLine()
    
    sql = 'DROP TABLE IF EXISTS ' + DBSettings.Table 
    _ = engine.execute(sql)
    
    print('Inserting rows: ', df.shape[0], 'into table: ', DBSettings.Table)
    
    df.to_sql(DBSettings.Table, engine, index = False, if_exists = 'append', chunksize = 4)
    
    engine.dispose()

    
def main():
    '''
    Start script with given parameters: messages_filepath, categories_filepath, database_filepath
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        if DBSettings.UseConfig:
            database_filepath = DBSettings.Database
        
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