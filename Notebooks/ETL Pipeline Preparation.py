
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# In[1]:


# import libraries
import DataScienceHelperLibrary as dsh
import pandas as pd
import numpy as np

from sqlalchemy import create_engine


# In[2]:


# One function in my library class: series.str.replace(... regex = False) 
# The version I tested here on this server throw an error saying 'regex' is unknown parameter.
# So updating fixed this error. 

#!pip install --upgrade pandas   


# In[3]:


_, files = dsh.ReadCsvFiles(['messages.csv', 'categories.csv'])


# In[4]:


# load messages dataset
messages = files['messages.csv']
messages.head()


# In[5]:


dsh.AnalyzeDataFrame(messages)


# As there are many rows with no original text, let's check what's the difference between those with orig text and those without:

# In[6]:


messages[messages['original'].isnull()].head()


# In[7]:


messages[~messages['original'].isnull()].head()


# I assume that the column 'original' keeps the message original text if it is not written in english. If the oiginal text is written in english, it is stored in the message column and the value in the corresponding original cell is NaN (or NULL in database).
# 
# 
# Messages starting with 'NOTES:' got my attention, so have a look at them.

# In[8]:


ser = messages['message']

ser[ser.str.startswith('NOTES:')]


# Most of the messages are like 'Not important', 'Already translated' and so on.
# So I decide to drop those rows.

# In[9]:


dsh.AnalyzeColumn(messages, 'genre')


# In[10]:


# load categories dataset
categories = files['categories.csv']
categories.head()


# In[11]:


dsh.AnalyzeDataFrame(categories)


# In[12]:


dsh.AnalyzeEqualColumns(messages, categories)


# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps

# In[13]:


# merge datasets
df = dsh.QuickMerge(messages, categories)

dsh.DfTailHead(df, 4)


# In[14]:


print(df.shape)


# Now there are more rows due to duplicate entries in both files.

# In[15]:


dsh.DfTailHead(messages[~messages['original'].isnull()], 4)


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.

# In[16]:


uniqueCategories = dsh.GetUniqueValuesListFromColumn(
    df, 
    'categories', 
    clean = {'' : ['-1', '-0'] },
    splitby = ';',
)


# In[17]:


_ = dsh.CheckIfValuesContainedInEachOther(uniqueCategories)


# I made this check because I was thinking about an alternative way to encode this column. Just extract the pure category name and check if "{category}-1" in whole category string or not. Depending on that add column with values 1 or 0. But that won't work with 'related' as the output says.

# The last value 'related-2' attracted my attention. I assumed that it is in the first column. So I checked it:

# In[18]:


df[df['categories'].str.contains('related-2')]


# My first asumption - without having a look at the messages - was that higher numbers are something like a weight (0 does not belong to this category, 1 tendentially/probably yes and 2 definitely. 
# But as I saw the messages, my assumption "english text in message column, foreign text - if present, in original column".
# 
# There are multiple ways how to treat multiple languages and I decided myself for the bold printed one:
# - store in database and use for model,
# - #### store in database to have data present and select via sql only those with related = 1 or just drop those rows before training,
# 
# - either, or... don't consider and just drop.

# In[19]:


uniqueCategories.remove('related-2')


# In[20]:


ser = df['categories']

categories = ser.str.split(';', expand = True)


# In[21]:


# select the first row of the categories dataframe
#row = df[:1]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing

category_colnames = uniqueCategories
print(category_colnames)


# In[22]:


# rename the columns of `categories`

categories.columns = category_colnames


# In[23]:


categories.head(n = 2)


# ### 4. Convert category values to just numbers 0 or 1.
# - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
# - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

# In[24]:


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


# In[25]:


categories.head(n = 2)


# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.
# - Concatenate df and categories data frames.

# In[26]:


# For Testing
dfcopy = df.copy(deep = True)


# In[27]:


# drop the original categories column from `df`
df = df.drop('categories', axis = 1)

df.head()


# In[28]:


print(df.shape, categories.shape)


# In[29]:


# concatenate the original dataframe with the new `categories` dataframe

#dfFinal = pd.concat([df, categories], axis = 1, sort = None )

dfFinal = df.join(categories, how = 'inner', on = 'id')

dfFinal.head()


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# In[30]:


# check number of duplicates
# drop duplicates
# check number of duplicates

# These steps are implemented in following function:

dfFinal = dsh.RemoveDuplicateRows(dfFinal)


# As my analysis shows (after loading the file), there are still NaN values in the dataframe.
# So before saving values in database, I replace Nan by None (in hope it will be converted to DBNULL).

# In[31]:


dfFinal.isnull().sum()


# In[32]:


dfFinal[dfFinal.original.isnull()]


# ## Remark:
# 
# As there is only the column 'original' with nan values, we can just replace them by None so that it is a clear representation in database.

# In[33]:


dfDb = dfFinal #.replace(np.nan, None)


# In[34]:


dsh.AnalyzeNanColumns(dfDb)


# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.

# ## Remark:  
# "Clean" is relative. I just encoded and removed duplicates from matching duplicate ids from categories to duplicate ids from messages (as explored above).
# 
# I was deliberating to "clean" data for database by removing rows starting with 'NOTES:' and so on. But I decided against this step because it might be that somehow values appear in database that did not go through that cleaning routine. So I am going to analyze and drop rows that are irreleveant/not processable before using for modeling.

# In[35]:


database = 'MyDisasterResponse.db'
table = 'Message'

useOrig = False

#if not useOrig:
#    origDb = 'My' + origDb
#    origMessages = 'My' + origMessages

engine = create_engine('sqlite:///{}'.format(database))


# In[36]:


dsh.PrintLine('Current Tables in DB:')
print(engine.table_names())
dsh.PrintLine()


# In[37]:


sql = 'DROP TABLE IF EXISTS ' + table 
_ = engine.execute(sql)


# In[38]:


print('Inserting rows: ', dfFinal.shape[0], 'into table: ', table)


# In[39]:


dfDb.to_sql(table, engine, if_exists = 'append', chunksize = 4)


# In[40]:


engine.dispose()


# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.

# In[41]:


#!tar chvfz ETL_Pipeline_Preparation.tar.gz *

