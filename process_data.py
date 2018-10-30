import DataScienceHelperLibrary as dsh
import pandas as pd
import numpy as np
import sys

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
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
        categories[column] = categories[column].apply(lambda x: int(str(x).replace(column + '-', "")))
        # convert column from string to numeric

        # This step does not need to be done beuase I'm already calling int(..) in apply
        typ = str(categories[column].dtype)
        if typ in typelist:
            continue
        typelist.append(typ)
    print('DType/s of new columns is/are: ', typelist)
    
    df = df.drop('categories', axis = 1)
    df = df.join(categories, how = 'inner', on = 'id')
    
    df = dsh.RemoveDuplicateRows(df)
    dsh.AnalyzeNanColumns(df)
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    table = 'Message'
    
    dsh.PrintLine('Current Tables in DB:')
    print(engine.table_names())
    dsh.PrintLine()
    
    sql = 'DROP TABLE IF EXISTS ' + table 
    _ = engine.execute(sql)
    
    print('Inserting rows: ', df.shape[0], 'into table: ', table)
    
    df.to_sql(table, engine, if_exists = 'append', chunksize = 4)
    
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