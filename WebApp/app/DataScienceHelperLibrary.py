"""Small Library for very common uses.

This module provides parameterizable functions 
for analyzing data frames with log output.
"""

__version__ = '0.2'
__author__ = 'Benjamin Wenner'


from encodings.aliases import aliases
from multiprocessing import Pool
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from sqlalchemy import create_engine

import fnmatch as fnmatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb


import glob
import os
import string
import datetime
import sys

# Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


def IsMatch(txt, wildcard):
    '''
    Check if text contains certain subtext by wildcard.
    
    INPUT:
    txt: string: text to search
    wildcard: string: wildcard applied to text
    '''
    return fnmatch.fnmatch(txt, wildcard)

def PrintLine(text = '-', number = 20, character = '-'):
    print(character * number, text, character * number)

def DfTailHead(df, count = 15):
    '''
    Returns concatination from dataframes head and tail.
    
    INPUT:
    df: Dataframe
    count: int: number of rows for both head and tail
    
    OUTPUT:
    dataframe: concatination of both
    '''
    count = min(abs(count) if count != 0 else 15, df.shape[0])
    return pd.concat([df.head(count), df.tail(count)])

def IsNullOrEmpty(text):
    return text is None or len(text) == 0

def KeepLetters(text):
    '''
    Removes all characters that are not alpha.
    '''
    if text is None or text is not str:
        return text
    return ''.join(s for s in text if s.isalpha())

def KeepLettersNumbers(text):
    '''
    Removes all characters that are not alphanumeric.
    '''
    if text is None:
        return text
    return ''.join(s for s in text if s.isalnum())

def CheckIfValuesContainedInEachOther(values):
    probdir = {}
    for i1, v1 in enumerate(values):
        probs = []
        for i2, v2 in enumerate(values):
            if i1 >= i2:
                continue
            if v1 in v2 or v2 in v1:
                probs.append(v2)
        if len(probs) > 0:
            probdir[v1] = probs
    if len(probdir.keys()) > 0:
        PrintLine('Following values are contained in others:')
        for key in list(probdir.keys()):
            print(key, ' - ', probdir[key])
        PrintLine()
    else:
        PrintLine('No values are contained in others')
    return len(probdir.keys()) > 0
 
def GetAsList(element):
    '''
    INPUT:
    element: float, int, string, list, set, tuple, ndaray
    
    OUTPUT:
    returns [x] for string/number, list[x] for list types, 
    list[x.values] for ndarray
    '''
    if str(type(object)) in [
        'float', 'float32', 'float64', "<class 'float'>"
        'int', 'int32', 'int64', "<class 'int'>", 
        'str', "<class 'str'>" 
    ]:
        return [object]
    if type(object) == list:
        return object
    if type(object) == set or type(object) == tuple:
        return list[object]
    if type(object) == np.ndarray:
        return list(object.values)
    raise ValueError('Type unknown: ', type(object))
    
def AnalyzeColumn(df, column, analyzeNan = True, analyzeVc = True):
    '''
    INPUT:
    df: Dataframe
    column: column name or list of names to Analyze
    analyzeNan: bool
    analyzeVec: bool
    '''
    if not type(column) is list:
        column = [column]    
    PrintLine('Analysing column/s "{}"'.format(column))
    for col in column:
        print('Datatype (dtype) = ', df[col].dtype)
        if analyzeNan:
            AnalyzeNanColumns(df, col)
        if AnalyzeValueCounts:
            AnalyzeValueCounts(df, columns = col)
    PrintLine('Finished analysing column/s "{}"'.format(column))

def AnalyzeNanColumns(df, columns = None):
    '''
    INPUT:
    df: Dataframe
    columns = str or list
    '''
    if df is None:
        raise ValueError('Fnc "AnalyzeNanColumns": df is None')
    if columns is not None:
        columns = GetAsList(columns)
    else:
        columns = list(df.columns)
    PrintLine('Analysis of Columns with NaN values')

    dfnull = df[columns].isnull().mean()
    
    if dfnull.shape[0] == 0:
        print('All columns have values')
        PrintLine('Analysis of Columns with NaN values finished')
        return
    renameDic = {}
    for ind, index in enumerate(dfnull.index):
        renameDic[dfnull.index[ind]] = str(index) + ', type: ' + str(df[index].dtype)
    dfnull = dfnull.rename(renameDic)
    tmp = dfnull[dfnull == 0]
    if tmp.shape[0] > 0:
        print('Columns having all values: {0}, {1:.2f}%'.format(len(tmp), len(tmp) * 100 / len(columns)))
        print(tmp)
    tmp = dfnull[(dfnull > 0) & (dfnull <= 0.05)]
    if tmp.shape[0] > 0:
        print('Columns having > 0% and <= 5% missing values: {0}, {1:.2f}%'.format(len(tmp), len(tmp) * 100 / len(df.columns)))
        print(tmp)
    tmp = dfnull[(dfnull > 0.05) & (dfnull <= 0.2)]
    if tmp.shape[0] > 0:
        print('Columns having > 5% and <= 20% missing values: {0}, {1:.2f}%'.format(len(tmp), len(tmp) * 100 / len(df.columns)))
        print(tmp)
    tmp = dfnull[(dfnull > 0.2) & (dfnull <= 0.5)]
    if tmp.shape[0] > 0:
        print('Columns having > 20% and <= 50% missing values: {0}, {1:.2f}%'.format(len(tmp), len(tmp) * 100 / len(df.columns)))
        print(tmp)
    tmp = dfnull[(dfnull > 0.5) & (dfnull <= 0.7)]
    if tmp.shape[0] > 0:
        print('Columns having > 50% and <= 70% missing values: {0}, {1:.2f}%'.format(len(tmp), len(tmp) * 100 / len(df.columns)))
        print(tmp)
    tmp = dfnull[dfnull > 0.7]
    if tmp.shape[0] > 0:
        print('Columns having > 70% missing values: {0}, {1:.2f}%'.format(len(tmp), len(tmp) * 100 / len(df.columns)))
        print(tmp)
    PrintLine('Analysis of Columns with NaN values finished')
    

    
def AnalyzeValueCounts(df, columns = None, types = None, considerMaxValues = 20):
    '''
    INPUT:
    df: Dataframe
    columns = None or column name or list of columns
    types = None or columns with provided types
    considerMaxValues: Print values if # is <= xrange
    '''
    if df is None:
        raise ValueError('Fnc "AnalysisValueCounts": df is None')
    if (considerMaxValues < 0 or considerMaxValues > 30):
        raise ValueError('Fnc "AnalysisValueCounts": considerMaxValues < 0 or too large (> 30)', considerMaxValues)
    logtxt = 'Considering columns: '
    if columns is None or types is None:
        if columns is None and types is None:
            columns = list(df.columns)
        elif types is None and columns is not None:
            if not type(columns) is list:
                columns = [columns]
        elif types is not None:
            columnstmp = list(SelectColumnsByType(df, types).columns)
            if columns is None:
                columns = columnstmp
            else:
                columns = [col for col in columnstmp if col in columns]
            
    if len(columns) == 0:
        print('No columns to Analyze value counts for. Passed columns and types: ', columns, types)
        return
    print(logtxt, columns)
    PrintLine('Dataframe value counts analye started')
    colsWithOnlyOneValue = []
    for col in columns:
        PrintLine('', character = '*')
        vcser = df[col].value_counts()
        if vcser.shape[0] == 1:
            colsWithOnlyOneValue.append(col)
        if vcser.shape[0] > considerMaxValues:
            print('More than {} different values: '.format(considerMaxValues), vcser.shape[0])
            print('Name: ', col, ', dtype: ', vcser.dtype)
        else:
            print(vcser)
        PrintLine('', character = '*')
    if len(colsWithOnlyOneValue) > 0:
        PrintLine('There are columns with only one value: ', number = 10, character = '!')
        print(colsWithOnlyOneValue)
        PrintLine('', number = 10, character = '!')
    PrintLine('Dataframe value counts analysis finished')
    
    
    
def AnalyzeDataFrame(df):
    '''
    INPUT:
    df: Dataframe
    '''
    if df is None:
        raise ValueError('Fnc "AnalyzeDataframe": df is None')
    PrintLine('Dataframe analysis started')
    print('Shape: ', df.shape)
    
    print('Number of duplicate rows: ', df.shape[0] - df.drop_duplicates().shape[0])
    
    AnalyzeNanColumns(df)
    AnalyzeValueCounts(df)
    
    PrintLine('Dataframe analysis finished')

def GetEqualColumns(df1, df2):
    '''
    INPUT:
    df1, df2: DataFrame
    
    OUTPUT:
    List of equal columns or empty list, 
    list columns df1, 
    list columns df2 
    '''
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)
    ret = [ col for col in cols1 if col in cols2 ]
    if len(ret) > 0:
        print('Equal columns found: ', ret)
    else:
        print('No equal columns found')
    return ret, cols1, cols2
    
    
def AnalyzeEqualColumns(df1, df2):
    '''
    INPUT:
    df1, df2: DataFrame
    '''
    PrintLine('Starting comparing dataframes:')
    equals, cols1, cols2 = GetEqualColumns(df1, df2)
    
    if len(equals) > 0:
        for col in equals:
            uq1 = set(df1[col].unique())
            uq2 = set(df2[col].unique())
            
            if len(uq1 - uq2) == 0 and len(uq2 - uq1) == 0:
                print('Column {}: All values in both columns contained'.format(col))
            diff = uq1 - uq2
            if len(diff) > 0:
                print('Column {}: Values contained in df1 but not in df2: '.format(col), diff)
            diff = uq2 - uq1
            if len(diff) > 0:
                print('Column {}: Values contained in df2 but not in df2: '.format(col), diff)
        vc1 = df1.id.value_counts()
        vc2 = df2.id.value_counts()
        diff = [ (msgid, vc1[msgid], vc2[msgid]) for msgid in vc1.index if vc1[msgid] != vc2[msgid] ]
        if len(diff) > 0:
            print('Column {}: Value counts differ: '.format(col), diff)
        else:
            print('Column {}: Value counts are equal'.format(col))
    else:
        print('No equal column names found:')
        print(cols1)
        print(cols2)
    PrintLine('Finished comparing dataframes:')
    
def AppendColumnByValuesInCell(df, column, newcolumn, values):
    '''
    INPUT:
    df: Dataframe
    column: column to search values
    newcolumn: new column name
    values: value or list of values
    
    OUTPUT:
    returns dataframe with newcolumn where 1 means value from column
            contains an item from values
    '''
    dfcopy = df.copy(deep = True)
    dfcopy[newcolumn] = dfcopy[column].apply(lambda x: 1 if x is not np.NaN and any(val in x for val in values) else 0)
    return dfcopy
    
def Apply10Encoding(df, column, vals, newcol = None, drop = True):
    '''
    INPUT:
    df: Pandas Dataframe
    column: column to encde with 1 or 0
    vals: values to encode with 1
    newcol: new column name
    drop: if newcol provided, column will be dropped.
    
    OUTPUT:
    df: data frame with columns of type 'object'
    '''
    if df is None:
        raise ValueError('Fnc "Apply10Encoding": df is None')
    if type(vals) is not list:
        vals = [vals]
    _encode = lambda x: 1 if x in vals else 0 
    if newcol is None:
        newcol = column
        drop = False
    df[newcol] = df[column].apply(_encode)
    if drop:
        return df.dropna()
    return df

def ApplyOneHotEncodingOnColumnWithMultiValuesInCell(df, column, values, drop = True, ignoreEmpty = True):
    '''
    INPUT:
    df: Pandas Dataframe
    column: column name
    drop: Drop column
    
    OUTPUT:
    df: Dataframe with one hot encoded column
    '''
    if df is None:
        raise ValueError('Fnc "Apply10Encoding": df is None')
    if df[column].dtype != 'O':
        raise ValueErro('Fnc "Apply10Encoding": Invalid dtype "{}" for column "{}""'.format(df[columns].dtype, column))
    dfcopy = df.copy(deep = True)
    newcolumns = []
    PrintLine('Start applying one hot encoding for columns "{}" and values "{}"'.format(column, values))
    print('Columns before one hot encoding: ', dfcopy.shape[1])
    for val in GetAsList(values):
        newcol = column + '_' + val
        newcolumns.append(newcol)
        #dfcopy[newcol] = dfcopy[dfcopy[column].apply(lambda x: 1 if val in x else 0)]
        dfcopy[newcol] = dfcopy[column].str.contains(val) * 1
        if ignoreEmpty:
            val = list(dfcopy[newcol].unique())
            if len(val) == 1:
                dfcopy = dfcopy.drop(newcol, axis = 1)
                print('New column "', newcol, '" dropped, only this value present: ', val)
    if drop:
        dfcopy = dfcopy.drop(column, axis = 1)
    print('New columns are: ', newcolumns)
    PrintLine('Finished applying one hot encoding')
    return dfcopy    
    
def ApplyOneHotEncoding(df, columns, ignoreEmpty = True):
    '''
    INPUT:
    df: Pandas Dataframe
    columns: column name or list of columns
    OUTPUT:
    df, newcols: Dataframe with one hot encoded columns that were passed and list of new columns.
    '''
    if df is None:
        raise ValueError('Fnc "ApplyFillMissingValuesWithMean": df is None')
    if columns is None or len(columns) < 1:
        print('Fnc "OneHotEncodeColumns": columns is empty/None')
        return df
    if not type(columns) is list:
        columns = [columns]
    ohdf = df.copy(deep = True)
    colbefore = list(ohdf.columns)
    PrintLine('Start applying one-hot encoding on: {}'.format(columns))
    print('Columns before one hot encoding: ', df.shape[1])
    print('Columns to be removed and replaced: ', columns)
    ohdf = pd.get_dummies(ohdf, columns = columns)
    colsafter = list(ohdf.columns)
    print('Size after encoding: ', len(colsafter))
    colsremoved = []
    for col in columns:
        if not col in colsafter:
            colsremoved.append(col)
    if len(columns) == len(colsremoved):
        newcols = []
        for col in list(ohdf.columns):
            if col in colbefore or col in colsremoved:
                continue
            newcols.append(col)
        print('Columns successfully one hot encoded :)  New columns are: ', newcols)
    
    print('Removed columns after one hot encoding: ', colsremoved)
    PrintLine('Finished applying one-hot encoding')
    return ohdf, newcols

def ApplyFillMissingValuesWithMean(df):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    df: Dataframe and nan values filled with mean
    '''
    if df is None:
        raise ValueError('Fnc "ApplyFillMissingValuesWithMean": df is None')
    fill_na = lambda x: x.fill_na(x.mean())
    df.apply(fill_na, axis = 0)
    return df

    

def ConvertColumnToType(df, columns, newtype = 'float64', replace = None):
    '''
    INPUT:
    df: Pandas Dataframe
    columns: Column/s to convert
    newtype: Final type
    replace: Dictionary like { '': { "rep1", "rep2", ...} }
    
    OUTPUT:
    Dataframe with converted column
    '''
    if df is None:
        raise ValueError('Fnc "ConvertColumnToType": df is None')
    dfcopy = df.copy(deep = True)
    if type(columns) is not list:
        columns = [columns]
    PrintLine('Start replacing and converting columns')
    for col in columns:
        if replace is not None:
            if type(replace) is not dict:
                raise ValueError('Fnc "ConvertColumnToType": type of replace is no dictionary: ', replace)
            for key in replace.keys():
                if df[col].dtype == newtype:
                    print('Column "{}" dtype is already {}'.format(col, newtype))
                    break
                val = replace[key]
                for repval in val:
                    print('Replacing "{}" with "{}"'.format(repval, key))
                    dfcopy[col] = dfcopy[col].apply(lambda x: x.replace(repval, key).strip() if type(x) is str else x)
        dfcopy[col] = dfcopy[col].astype(newtype)
        print('New type of {} is: {}'.format(col, dfcopy[col].dtype))
    PrintLine('Replacing and converting columns finished')
    return dfcopy

def CleanValuesInColumn(df, columns, trim = True, clean = None):
    '''
    INPUT:
    df: Dataframe
    columns: column name or list of columns
    trim: remove leading and trailing spaces
    clean: dictionary<string, list<string>>: replace any string value in list with key
    
    OUTPUT:
    Dataframe with cleaned column/s
    '''
    if df is None:
        raise ValueError('Fnc "GetUniqueValuesFromColumn": df is None')
    dfcopy = df.copy(deep = True)
    if type(columns) is not list:
        columns = [columns]
    PrintLine('Start cleaning values in columns:')
    for col in columns:
        if dfcopy[col].dtype == 'O':
            applied = ''
            if clean is not None:
                for key in clean.keys():
                    repvalues = clean[key]
                    if len(repvalues) == 1:
                        dfcopy[col] = dfcopy[col].str.replace(repvalues[0], key)
                        applied = 'replaced "{}" by "{}"'.format(repvalues, key)
                    elif len(repvalues) == 2:
                        dfcopy[col] = dfcopy[col].str.replace(repvalues[0], key).str.replace(repvalues[1], key)
                        applied = 'replaced "{}" by "{}"'.format(repvalues, key)
                    elif len(repvalues) == 3:
                        dfcopy[col] = dfcopy[col].str.replace(repvalues[0], key).str.replace(repvalues[1], key).str.replace(repvalues[2], key)
                        applied = 'replaced "{}" by "{}"'.format(repvalues, key)
                    elif len(repvalues) == 4:
                        dfcopy[col] = dfcopy[col].str.replace(repvalues[0], key).str.replace(repvalues[1], key).str.replace(repvalues[2], key).str.replace(repvalues[3], key)
                        applied = 'replaced "{}" by "{}"'.format(repvalues, key)
                    else:
                        print('Number of replace values not supported: ', len(repvalues))
            if trim:
                dfcopy[col] = dfcopy[col].apply(lambda x: x.strip() if x is not np.NaN else x)
                if len(applied) > 0:
                    applied = applied + ', '
                applied = applied + 'trimmed'
            if len(applied) > 0:
                print('Applied on "{}": '.format(col), applied)
        else:
            print('Cannot apply cleaning on column "{}", dtype is: '.format(col, dfcopy[col].dtype))
    PrintLine('Finished cleaning values in columns:')
    return dfcopy
    
def CountMissingValuesInColumn(df, column):
    '''
    INPUT:
    df: Dataframe
    column: Column/s to count values
    
    OUTPUT:
    dfs: list of missing values
    '''
    if df is None:
        raise ValueError('Fnc "CountMissingValuesInColumn": df is None')
    if type(column) is str:
        if len(column) == 0:
            raise ValueError('Fnc "CountMissingValuesInColumn": column name is empty')
        column = [column]
    if column is None or len(column) < 0:
        raise ValueError('Fnc "CountMissingValuesInColumn": files is None or empty')
    results = []
    dfnull = df.isnull().sum()
    for col in column:
        results.append(dfnull[col])
    return results

'''
def ReadCsvFiles(directory, wildcards, extract = False, delimeter = ','):
    
    INPUT:
    directory: directory relative to notebook
    wildcards: List of file names
    
    OUTPUT:
    dfs: overall status (bool) all succ, list of data frames
'''
'''    
    if directory is None or len(directory) == 0:
        raise ValueError('Fnc "ReadCsvFiles": directory is None/empty')
    if wildcards is None or len(wildcards) == 0:
        raise ValueError('Fnc "ReadCsvFiles": wildcards is None/empty')
    wildcards = GetAsList(wildcards)
    try:
        files = []
        for wc in wildcards:
            
    
    except:
        raise ValueError('Fnc "ReadCsvFiles": df is None')
'''
    
def ReadCsvFiles(files, delimiter = ','):
    '''
    INPUT:
    files: List of file names
    
    OUTPUT:
    dfs: overall status (bool) all succ, list of data frames
    '''
    if files is None or len(files) < 0:
        raise ValueError('Fnc "ReadCsvFiles": files is None or empty')
    if type(files) is not list:
        files = [files]
    dfs = {}
    PrintLine('Start reading files')
    notworked = []
    for file in files:
        try:
            curdf = pd.read_csv(file, delimiter = delimiter)
            dfs[file] = curdf
            print('Dataframe loaded from {}: shape = {}'.format(file, curdf.shape))
        except Exception as e:
            print('Could not read file ', file, ': ', str(e))
            notworked.append(file)
            dfs[file] = None
    log = 'Reading files successfully finished'
    
    if len(notworked) > 0:
        for file in notworked:
            print('Trying to load file with encodings: ', file)
            for encoding in set(aliases.values()):
                try:
                    curdf = pd.read_csv(file, encoding = encoding)
                    dfs[file] = curdf
                    print('Encoding found to load file: ', encoding)
                    break
                except:
                    pass
    allsucc = True
    for key, val in enumerate(dfs):
        if val is None:
            print('File could not be loaded: ', key)
            allsucc = False
    PrintLine(log)
    return allsucc, dfs 
    
def GetColumnsHavingNan(df):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    ret: returns list of columns having nan values
    '''
    if df is None:
        raise ValueError('Fnc "GetColumnsHavingNan": df is None')
    return df.isnull()

    
def GetColumnHavingNanPercent(df, percent):
    '''
    INPUT:
    df: Pandas Dataframe
    percent: element from [0,1]
    OUTPUT:
    lst: returns dataframe of columns having more than 0.x missing values
    '''
    if df is None:
        raise ValueError('Fnc "GetColumnsHavingNanPercent": df is None')
    if percent > 1 or percent < 0:
        raise ValueError('Fnc "GetColumnsHavingNanPercent": percent is out of bounds [0,1]: ', percent)
    return df[df.columns[df.isnull().mean() > percent]]

    
def GetColumnsHavingNoNan(df):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    lst: returns list of columns having no nan values
    '''
    if df is None:
        raise ValueError('Fnc "GetColumnsHavingNoNan": df is None')
    return df[~df.isnull().any()]
    #return df[~df.isnull().mean() == 0]


    
def GetPropInGroupB(df, group, prop = None):
    '''
    INPUT:
    df: Pandas Dataframe
    prop: Column whose proportion we are looking for after grouping by column group
    group: Column the data will be grouped by to calc proportion for column prop
    
    OUTPUT:
    proportion: proportion of column prop in each group grouped by column group (for example group by employment
    status and get mean for job satisfaction)
    '''
    if df is None:
        raise ValueError('Fnc "GetPropAInGroupB": df is None')
    if not prop in df.columns:
        raise ValueError(str.format('Fnc "GetPropAInGroupB": prop: column "{}" not in dataframe', prop))
    if not group in df.columns:
        raise ValueError(str.format('Fnc "GetPropAInGroupB": group: column "{}" not in dataframe', group))
    dfmean = df.groupby(group).mean()
    if prop is None:
        return dfmean
    return dfmean[prop]

def GetUniqueValuesListFromColumn(df, column, trim = False, clean = None, splitby = None, asc = None, ignoreempty = False):
    '''
    INPUT:
    df: Dataframe
    column: Column name
    trim: Trim values
    clean: Dictionary to clean values before split.
    split: String to split cell value. For example: "A,B,C" -> ["A", "B", "C"]
    asc: Define sorting
    ignoreempty: Empty values are excluded from result
    
    OUTPUT:
    List of unique values
    '''
    if df is None:
        raise ValueError('Fnc "GetUniqueValuesFromColumn": df is None')
    if type(column) is not str:
        raise ValueError('Fnc "GetUniqueValuesFromColumn": column is not string: "', column, '"')
    dfcopy = None
    if trim or clean is not None:
        dfcopy = CleanValuesInColumn(df, column, trim, clean)
    else:
        dfcopy = df.copy(deep = True)
    vals = dfcopy[column].unique()
    finalVals = []
    if splitby is None:
        finalVals = vals
    elif dfcopy[column].dtype == 'O':
        for val in vals:
            splitted = val.split(splitby)
            for splitvalue in splitted:
                if splitvalue in finalVals:
                    continue
                finalVals.append(splitvalue)
    if trim:
        for ind in range(len(finalVals)):
            if type(finalVals) is not str:
                continue
            finalVals[ind] = finalVals[ind].strip()
    if ignoreempty:
        finalVals = [x for x in finalVals if type(x) is str and len(x) > 0]
    if asc is not None:
        if type(asc) is bool:
            finalVals.sort(reverse = asc == False)
    PrintLine('Column "{}" has {} unique values'.format(column, len(finalVals)))
    return finalVals

def PlotHeatmap(df, method = 'spearman', square = True, vmax = 1.0):
    '''
    INPUT:
    df: Dataframe
    strategy: Strategy for computation
    axis: Axis/orientation
    
    OUTPUT:
    Dataframe with imputedp9 columns that were passed.
    '''
    if df is None:
        raise ValueError('Fnc "PlotHeatmap": df is None')
    corMatData = df.copy(deep = True)
    corMat = corMatData.corr(method = method)
    fig, ax = plt.subplots()
    sb.heatmap(corMat, square = square, vmax = vmax)
    plt.show()
    return corMat
    
    
def ImputeNanValues(df, impute = 'NaN', strategy = 'median', axis = 0):
    '''
    INPUT:
    df: Dataframe
    strategy: Strategy for computation
    axis: Axis/orientation
    
    OUTPUT:
    Dataframe with imputedp9 columns that were passed.
    '''
    if df is None:
        raise ValueError('Fnc "NormalizeColumns": df is None')
    dfcopy = df.copy(deep = True)
    imputer = preprocessing.Imputer( missing_values = impute, strategy = strategy, axis = axis)
    impar = imputer.fit_transform(dfcopy)
    dfimp = pd.DataFrame(impar, columns = list(dfcopy.columns))
    return dfimp

def GetCommonColumns(df1, df2):
    '''
    Returns a list of identical column names
    
    INPUT:
    df1, df2: Dataframe
    
    OUTPUT:
    list of equal column names
    '''
    return [str(col) for col in list(df1.columns) if str(col) in list(df2.columns)]
    
def MergeFrames(df1, df2, how = 'inner', on = None):
    '''
    INPUT:
    df1, df2: Dataframe
    
    
    OUTPUT:
    returns merged dataframe
    '''
    if on is None or len(on) == 0:
        on = GetCommonColumns(df1, df2)
        if len(on) == 0:
            raise ValueError('Fnc "MergeFrames": no equal column names found')
    PrintLine('Dataframes merge: {}, {}'.format(how, str(on)))
    return df1.merge(df2, how = how, on = on)
    
def MinMaxOfSeries(serie):
    '''
    Returns minimum and maximum from serie
    '''
    return min(serie), max(serie)

def NormalizeSeries(serie):
    '''
    Normalize a serie by min and max value
    '''
    s_min, s_max = MinMaxOfSeries(serie)
    return (serie - s_min) / (s_max - s_min)


def NormalizeColumns(df, columns = None, newCols = None):
    '''
    INPUT:
    df: Dataframe
    columns: string or list of strings
    
    OUTPUT:
    Dataframe with normalized columns that were passed.
    '''
    if df is None:
        raise ValueError('Fnc "NormalizeColumns": df is None')
    dfcopy = df.copy(deep = True)
    if columns is None:
        columns = list(dfcopy.columns)
    else:
        columns = GetAsList(columns)
    if newCols is None:
        newCols = columns
    else:
        newCols = GetAsList(newCols)
    PrintLine('Start normalizing ', len(columns), ' columns')
    for ind, col in enumerate(columns):
        meanBefore = dfcopy[col].mean()
        dfcopy[newCols[ind]] = NormalizeSeries(dfcopy[col])
        meanAfter = dfcopy[col].mean()
        print('Normalized column {0}: mean changed from {1} to {2}'.format(newCols[ind], meanBefore, meanAfter))
    print('Finished normalizing columns')
    return dfcopy


def ReduceDimensions_PCA(df, n_comp = None, columns = None):
    '''
    INPUT:
    df: Pandas Dataframe
    n_comp: Number of pca components
    columns: possible list of columns. If None, all columns are used
    OUTPUT:
    df: Dataframe whose rows all have values
    '''
    if df is None:
        raise ValueError('Fnc "ReduceDimensions_PCA": df is None')
    if columns is None:
        columns = list(df.columns)
    if n_comp is None:
        n_comp = len(df.columns)
    dfcopy = df[columns].copy(deep = True)
    _pca = PCA(n_components = n_comp)
    dfcopy = _pca.fit_transform(dfcopy)
    return pd.DataFrame(dfcopy), _pca
    
def PCAPlotExplainedVariances(pca, features = None):
    '''
    INPUT:
    pca: PCA components
    features: Number of features to analyze
    '''
    variances = [val / 100 for val in pca.explained_variance_]
    if features is None or features <= 0:
        features = len(variances)
    plt.title('Analysis of principal components')
    plt.ylabel('Explained Variance')
    plt.xlabel('Number of Components')
    _ = plt.bar(range(0, features), variances[:features])
    
def PCAPlotCumulatedVariances(pca, features = None):
    '''
    INPUT:
    pca: PCA components
    features: Number of features to analyze
    '''
    variances = [val / 100 for val in pca.explained_variance_]
    if features is None or features <= 0:
        features = len(variances)
    plt.title('Analysis of principal components')
    plt.ylabel('Cumulated Explained Variance')
    plt.xlabel('Number of Components')
    _ = plt.plot(range(0, features), np.cumsum(variances)[:features])

def RemoveAllRowsHavingAnyMissingValue(df, log = True):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    df: Dataframe whose rows all have values
    '''
    if df is None:
        raise ValueError('Fnc "RemoveAllRowsHavingAnyMissingValue": df is None')
    colstoremove = GetColumnsHavingNan(df)
    if log:
        for remcol in colstoremove:
            print('Column will be removed from dataframe: ', remcol)
    return df.dropna()


    
def AnalyzePCAResult(pca, pcaind, df, text = False, consider = 0.08):
    '''
    INPUT:
    pca: PCA
    pcaind: index of component to Analyze and plot
    text: print components by absolute weight descending
    consider: plot components with absolute weight >= x
    '''
    dfpcp = pd.DataFrame(pca.components_[pcaind], index = df.columns)
    dfpcp['ordered'] = np.abs(dfpcp)
    dfpcp = dfpcp.sort_values('ordered', axis = 0, ascending = False)
    rng = int(len(df.columns))
    if (text):
        for ind in range(rng):
            print(dfpcp.index[ind], ': ', dfpcp.values[ind])
    
    dfpcp = dfpcp.drop('ordered', axis = 1)
    
    rng = int(rng / 3)
    
    collist = [np.random.rand(3,) for i in range(rng)]
    consider = abs(consider)
    dfpcp = dfpcp[(dfpcp[0] > consider) | (dfpcp[0] < -consider)]
    dfpcptp = dfpcp.transpose()
    ax = dfpcptp.plot(kind = 'bar', color = collist)
    ax.figure.set_size_inches(int(rng / 3), 8)
    ax.set_title('PCA component # ' + str(pcaind + 1))
    
def RemoveColumnsByPercent(df, percent):
    '''
    INPUT:
    df: Dataframe
    percent: number between 0 and 1
    
    OUTPUT:
    Dataframe without those columns
    '''
    if df is None:
        raise ValueError('Fnc "RemoveColumnsByPercent": df is None')
    dfcopy = df.copy(deep = True)
    cols2remove = GetColumnHavingNanPercent(dfcopy, percent)
    PrintLine('Start finding columns with % missing values >= {}'.format(percent * 100))
    if cols2remove.shape[0] > 0:
        print('Columns will be removed: ', list(cols2remove.columns))
    else:
        print('No columns found')
    PrintLine('Finished finding columns')
    return dfcopy[[col for col in dfcopy.columns if col not in cols2remove]]

    
    
def RemoveColumnsByWildcard(df, wildcards):
    '''
    INPUT:
    df: Dataframe
    wildcards: string or list of strings
    
    OUTPUT: 
    Dataframe without columns
    '''
    if df is None:
        raise ValueError('Fnc "RemoveColumnsByWildcard": df is None')
    if wildcards is None or len(wildcards) < 1:
        raise ValueError('Fnc "RemoveColumnsByWildcard": No wildcards passed: ', wildcards)
    rem = []
    if type(wildcards) is not list:
        wildcards = [wildcards]
    allColumns = list(df.columns)
    PrintLine('Start finding and removing columns matchting to wildcards: {}'.format(wildcards))
    dfcopy = df.copy(deep = True)
    for col in allColumns:
        for wc in wildcards:
            if IsMatch(col, wc):
                rem.append(col)
                break
    keep = [ ac for ac in allColumns if ac not in rem ]
    if len(rem) == 0:
        print('No column names found matchting to wildcards')
    else:
        print('Columns found to remove: ', rem)
    PrintLine('Finished removing columns matchting to wildcards')
    return dfcopy[ keep ]
    

def RemoveColumnsHavingOnlyOneValue(df):
    '''
    INPUT:
    df: Dataframe
    
    OUTPUT: 
    Dataframe without columns
    '''
    if df is None:
        raise ValueError('Fnc "RemoveColumnsHavingOnlyOneValue": df is None')
    PrintLine('Start searching and removing columns with one value:')
    cols = []
    keep = []
    for col in list(df.columns):
        vc = df[col].value_counts()
        if vc.shape[0] == 1 and vc[vc.index[0]] == df.shape[0]:
            print('Removing {} - "{}"'.format(col, vc.index[0]))
            cols.append(col)
            continue
        keep.append(col)
    dfret = df[keep]
    PrintLine('finished searching and removing columns with one value:')
    return dfret
   
def RemoveDuplicateRows(df):
    '''
    INPUT:
    df: Dataframe
    '''
    PrintLine('Removing duplicate rows')
    print('Current shape: ', df.shape)
    dupCnt = df.shape[0] - df.drop_duplicates().shape[0]
    if dupCnt == 0:
        print('No duplicates in dataframe')
        PrintLine()
        return df
    print('There are ', dupCnt, ' duplicates in the data')
    df = df.drop_duplicates()
    dupCnt = df.shape[0] - df.drop_duplicates().shape[0]
    if dupCnt == 0:
        print('All duplicates successfully removed. New size: ', df.shape)
    else:
        print('Could not remove all duplicates. Still remaining: ', dupCnt)
    PrintLine()
    return df
    
    
def RemoveRowsWithAllMissingValues(df, subset = None):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    df: Dataframe without rows with all features = nan
    '''
    if df is None:
        raise ValueError('Fnc "RemoveRowsWithAllMissingValues": df is None')
    return RemoveRowsByThresh(df, 1, subset)

def RemoveRowsByThresh(df, thresh, subset = None):
    '''
    INPUT:
    df: Pandas Dataframe
    thresh: Require that many non-NA values.
    subset: Columns to consider. If None, all columns are considered.
    
    OUTPUT:
    df: Dataframe with rows having at least 'thresh' values <> nan
    '''
    if df is None:
        raise ValueError('Fnc "RemoveRowsByThresh": df is None')
    dfcopy = df.copy(deep = True)
    if subset is not None and type(subset) is not list:
        subset = [subset]
    dfcopy = dfcopy.dropna(thresh = thresh, subset = subset)
    PrintLine('Rows removed by thresh = "{}": {}'.format(thresh, df.shape[0] - dfcopy.shape[0]))
    return dfcopy

def RemoveRowsWithValueInColumn(df, column, values, option = None):
    '''
    INPUT: 
    df: Dataframe
    column: string or collection of strings
    values: values to search in column. If row with value found, it will be removed.
    option: 'startswith', 'contains'
    
    OUTPUT:
    Dataframe without rows having values in column
    '''
    if df is None:
        raise ValueError('Fnc "RemoveRowsWithValueInColumn": df is None')
    if type(values) is not list:
        values = [values]
    PrintLine('Start removing rows')
    if option is not None:
        if option == 'startswith':
            for val in values:
                dfret = df[~df[column].str.startswith(val)]
        elif option == 'contains':
            for val in values:
                dfret = df[~df[column].str.contains(val)]
        else:
            raise ValueError('Fnc "RemoveRowsWithValueInColumn": option is invalid: ', option)
    else:
        dfret = df[~df[column].isin(values)]
    print('{} rows (ca. {}%) have been removed having value/s "{}" in column "{}"'.format(df.shape[0] - dfret.shape[0], "{0:.2f}".format((df.shape[0] - dfret.shape[0]) * 100 / df.shape[0]), values, column))
    print('New shape: ', dfret.shape)
    PrintLine()
    return dfret

def RemoveRowsByValuesOverAverage(df, column, times = 6):
    '''
    INPUT:
    df: DataFrame
    column: Column in df
    times: Number: if column_mean * times < cell then drop row
    '''
    if df is None:
        raise ValueError('Fnc "RemoveRowsByValuesOverAverage": df is None')
    mean = times
    dfret = None
    PrintLine('Start dropping rows with value/textlength > ' + str(times) + ' * column average')
    if df[column].dtype == 'O':
        mean = mean * df[column].str.len().mean()
        dfret = df[df[column].str.len() < mean]
    else:
        mean = mean * df[column].mean()
        dfret = df[df[column] < mean]
    print('Rows removed: ', df.shape[0] - dfret.shape[0])
    print('New shape: ', dfret.shape)
    PrintLine('Finished removing')
    return dfret
    
def SelectRowsWithValueInColumn(df, column, values, option = None):
    '''
    INPUT: 
    df: Dataframe
    column: string or collection of strings
    values: values to search in column..
    option: 'startswith', 'contains'
    
    OUTPUT:
    Dataframe with rows having values in column
    '''
    if df is None:
        raise ValueError('Fnc "SelectRowsWithValueInColumn": df is None')
    if type(values) is not list:
        values = [values]
    if option is not None:
        if option == 'startswith':
            for val in values:
                dfret = df[df[column].str.startswith(val)]
        elif option == 'contains':
            for val in values:
                dfret = df[df[column].str.contains(val)]
        else:
            raise ValueError('Fnc "RemoveRowsWithValueInColumn": option is invalid: ', option)
    else:
        dfret = df[df[column].isin(values)]
    print('{} rows (ca. {}%) have been removed not having value/s "{}" in column "{}"'.format(df.shape[0] - dfret.shape[0], "{0:.2f}".format((df.shape[0] - dfret.shape[0]) * 100 / df.shape[0]), values, column))
    print('New shape: ', dfret.shape)
    return dfret

def ScaleFrame(df, copy = True, withMean = True, withStd = True):
    '''
    INPUT: 
    df: Dataframe
    
    OUTPUT:
    Scaled dataframe (default with mean = 0 and std = 1 to avoid negative side effects from outliners)
    '''
    
    if df is None:
        raise ValueError('Fnc "ScaleValues": df is None')
    dfcopy = df.copy(deep = True)
    scaler = preprocessing.StandardScaler(copy = copy, with_mean = withMean, with_std = withStd)
    
    scaledar = scaler.fit_transform(dfcopy)
    
    return pd.DataFrame(scaledar, columns = list(dfcopy.columns)), scaler

def ScaleValues(X, y, copy = True, withMean = True, withStd = True):
    '''
    INPUT: 
    df: Dataframe
    
    OUTPUT:
    Scaled dataframe (default with mean = 0 and std = 1 to avoid negative side effects from outliners)
    '''
    scaler = preprocessing.StandardScaler(copy = copy, with_mean = withMean, with_std = withStd)
    _ = scaler.fit(X, y)
    XScaled = scaler.transform(X, y)
    
    return XScaled, scaler
    
def SplitDataInBinaryColumn(df, column):
    '''
    INPUT:
    df: Pandas Dataframe
    column: Column name
    
    OUTPUT:
    List of dataframes splitted by values
    '''
    if df is None:
        raise ValueError('Fnc "SplitDataByValuesInColumn": df is None')
    values = list(df[column].unique())
    if len(values) > 2:
        raise ValueError('Fnc "SplitDataByValuesInColumn": more than 2 values in column ', column)
    list1 = []
    list2 = []
    list3 = []
    list1.append(values[0])
    list2.append(values[1])
    list3.append(list1)
    list3.append(list2)
    return SplitDataByValuesInColumn(df, column, list3)
    
def SplitDataByValuesInColumn(df, column, values):
    '''
    INPUT:
    df: Pandas Dataframe
    column: Column name
    values: list of list of values. For example: [ [val1_A, val2_A], [val1_B, val2_B] ] 
    results in two dataframes. Values of passed column are in [val1_A, val2_A] for first data frame 
    and values for passed column are in [val1_B, val2_B] for seccond dataframe.
    
    OUTPUT:
    List of dataframes splitted by values
    '''
    if df is None:
        raise ValueError('Fnc "SplitDataByValuesInColumn": df is None')
    dflist = []
    log = 'Start splitting data by values ' + str(values) + ' in column: ' + str(column)
    PrintLine(log)
    for valueList in values:
        newdf = df[df[column].isin(valueList)]
        print('New dataframe, based on value list', valueList, ', with shape: ', newdf.shape)
        dflist.append(newdf)
    PrintLine('Finished splitting data')
    return dflist

def SplitDataTrainTest(X, y, testSize = 0.3, randomState = 42):
    '''
    INPUT:
    X: Features
    y: Result
    
    returns XTrain, XTest, yTrain, yTest
    '''
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = testSize, random_state = randomState)
    PrintLine('Test/Train split (test size = {}, random state = {}:'.format(testSize, randomState))
    print('Training: X = {0}, y = {1}'.format(XTrain.shape, yTrain.shape))
    print('Test    : X = {0}, y = {1}'.format(XTest.shape, yTest.shape))
    PrintLine()
    return XTrain, XTest, yTrain, yTest
    
def SplitDataInXY(df, colx, coly):
    '''
    INPUT:
    df: Pandas Dataframe
    colx: list of columns for X
    coly: list of columns for y
    
    OUTPUT:
    data frames x, y splitted by colx, coly
    '''
    if df is None:
        raise ValueError('Fnc "SplitDataInXY": df is None')
    return df[colx], df[coly]
    

    
def SelectColumnsByTypeObject(df):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    df: data frame with columns of type 'object'
    '''
    return SelectColumnsByType(df, ['object'])
    
def SelectColumnsByType(df, typeinc, typeexc = None):
    '''
    INPUT:
    df: Pandas Dataframe
    typeinc: list of types to include
    typeexc: list of types to exclude (default none)
    
    OUTPUT:
    returns data frames x, y selected by colx, coly
    
    NOTES:
    To select all numeric types, use np.number or 'number'
    To select strings you must use the object dtype, but note that this will return all object dtype columns
    See the numpy dtype hierarchy
    To select datetimes, use np.datetime64, 'datetime' or 'datetime64'
    To select timedeltas, use np.timedelta64, 'timedelta' or 'timedelta64'
    To select Pandas categorical dtypes, use 'category'
    To select Pandas datetimetz dtypes, use 'datetimetz' (new in 0.20.0) or 'datetime64[ns, tz]'
    '''
    if df is None:
        raise ValueError('Fnc "SelectColumnsByType": df is None')
    if type(typeinc) is not list:
        typeinc = [typeinc]
    return df.select_dtypes(include = typeinc, exclude = typeexc)

def SelectColumnsByWildcard(df, wildcards, logfound = False):
    '''
    INPUT:
    df: Dataframe
    wildcards: string or list of strings
    
    OUTPUT: 
    Dataframe with columns
    '''
    if df is None:
        raise ValueError('Fnc "SelectColumnsByWildcard": df is None')
    if wildcards is None or len(wildcards) < 1:
        raise ValueError('Fnc "SelectColumnsByWildcard": No wildcards passed: ', wildcards)
    rem = []
    if type(wildcards) is not list:
        wildcards = [wildcards]
    allColumns = list(df.columns)
    PrintLine('Start finding and keeping columns matchting to wildcards: {}'.format(wildcards))
    keep = []
    for col in allColumns:
        for wc in wildcards:
            if IsMatch(col, wc):
                keep.append(col)
                break
    if len(keep) == 0:
        print('No column names found matchting to wildcards')
    elif logfound:
        print('Columns found to keep: ', keep)
    PrintLine('Finished keeping columns matchting to wildcards: {}'.format(len(keep)))
    return df[ keep ]


def SqlCheckValuesInTable(df, table, engine):
    '''
    INPUT:
    df: Dataframe
    table: string
    engine: Sqlalchemy engine
    
    OUTPUT:
    returns true or false
    '''   
   
def QuickMerge(df1, df2, how = 'inner'):
    '''
    Performs by default an inner join on columns with equal column names
    Raises ValueException if there are no equal columns
    
    INPUT:
    df1, df2: DataFrames
    
    OUTPUT:
    Joined tables
    '''
    possibleCols = GetCommonColumns(df1, df2)
    if len(possibleCols) == 0:
        raise ValueError('No common columns found to merge on')
    PrintLine('Merging dataframes: ')
    print('Joined dataframes on equal columns: ', possibleCols)
    df = df1.merge(df2, how = how, on = possibleCols)
    print('New shape: ', df.shape)
    PrintLine()
    return df

    

def TrainModel(model, XTest, yTest):
    '''
    INPUT:
    model: Model providing .fit()
    XTest: Test data (Numnpy array or Dataframe)
    yTest: test labelds (Numnpy array or Dataframe)
    '''
    if not callable(getattr(model, 'fit')):
        raise ValueError('Fnc "TrainModel": model has no callablemethod "fit"')
    X = XTest
    y = yTest
    if not type(XTest) == np.ndarray:
        #if type(XTest) == pd.DataFrame:
        #    print('XTest: Dataframe passed, using values')
        #    X = XTest.values
        #else:
            raise ValueError('Fnc "TrainModel": XTest is not ndarray')
    if not type(yTest) == np.ndarray:
        #if type(yTest) == pd.DataFrame:
        #    print('yTest: Dataframe passed, using values')
        #    y = yTest.values
        #else:
            raise ValueError('Fnc "TrainModel": yTest is not ndarray')
    PrintLine('Start fitting model to data')
    start = datetime.datetime.now()
    fitted = model.fit(X, y)
    PrintLine('Train time: ' + str(datetime.datetime.now() - start ))
    return fitted


def MultiClassifierScoreF1(yTest, yPred):
    """
    INPUT:
    yTest: Array of labels
    yPred: Array of predicted labels

    OUTPUT:
    score: Median of F1 scores for each output classifier
    """
    f1Scores = []
    for i in range(np.shape(yPred)[1]):
        f1 = f1_score(np.array(yTest)[:, i], yPred[:, i], average = None)
        f1Scores.append(f1)
        
    score = np.median(f1Scores)
    return score








    
##################################################









##################################################


'''
def SplitCategoricalValues(df, dirValTypes):
    categoricalColumns, needsToEncodeBin, needsToEncodeMulti, needsToEncodeStringBin, needsToEncodeStringMulti, ignoreColumns4Encoding = {}, {}, {}, {}, {}, []

    for col in dirValTypes['categorical']:
        if not col in df:
            continue

        dfind = df.columns.get_loc(col)

        valueCounts = df.iloc[:, dfind].value_counts()
        valCount = valueCounts.count()

        categoricalColumns[col] = valCount

        added = False
        for ax in valueCounts.axes:
            for val in ax.values:
                try:
                    nbr = int(val)

                    # if any other value appears it must be new encoded
                    if nbr == 0 or nbr == 1:
                        continue

                    if valCount == 2:
                        if not col in needsToEncodeBin:
                                needsToEncodeBin[col] = []
                        if (val in needsToEncodeBin[col]):
                            continue
                        needsToEncodeBin[col].append( val )
                    else:
                        if not col in needsToEncodeMulti:
                                needsToEncodeMulti[col] = []
                        if (val in needsToEncodeMulti[col]):
                            continue
                        needsToEncodeMulti[col].append( val )
                    added = True
                except:
                    if valCount == 2:
                        if not col in needsToEncodeStringBin:
                                needsToEncodeStringBin[col] = []
                        if (val in needsToEncodeStringBin[col]):
                            continue
                        needsToEncodeStringBin[col].append( val )
                    else:
                        if not col in needsToEncodeStringMulti:
                                needsToEncodeStringMulti[col] = []
                        if (val in needsToEncodeStringMulti[col]):
                            continue
                        needsToEncodeStringMulti[col].append( val )
                    added = True
        if not added:
            ignoreColumns4Encoding.append(col)

    return categoricalColumns, needsToEncodeBin, needsToEncodeMulti, needsToEncodeStringBin, needsToEncodeStringMulti, ignoreColumns4Encoding    

'''




