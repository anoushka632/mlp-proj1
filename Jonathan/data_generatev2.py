#%%
#os library version
from pathlib import Path

# Data libraries
import pandas as pd
import numpy as np
import re

# sklearn modules
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer


# extra modules
from schrutepy import schrutepy


d = pd.read_csv("../the_office.csv")
transcripts = schrutepy.load_schrute()

##string cols
str_cols = (transcripts.applymap(type) == str).all(0)

##Lower names, character's and lines for easier processing
transcripts = transcripts.applymap(lambda s: s.lower() if type(s) == str else s)

##Remove spaces in character (some are incorrectly written like "jim ")
transcripts['character'] = transcripts['character'].str.replace(" ", "")
transcripts['season_ep'] = transcripts['season'].astype(str) + "_" + transcripts['episode'].astype(str)

# ##Per Episode Character lines
# transcripts.character.unique()
line_sum = transcripts.groupby(["season_ep", "character"]).size().reset_index(name = "lines") ##per character and episode lines
line_sum = line_sum.sort_values(ascending = False, by = "lines")##sort by lines

##Total line percentage per character
line_perc = line_sum.groupby("character").agg({"lines": "sum"}).sort_values(ascending = False, by = "lines")
line_perc['percentage'] = line_perc/(line_perc.sum())

##Greater than 1% line share
main_char = line_perc.loc[line_perc['percentage'] > 0.01].index

##Top 10 line speakers (main_char)
# main_char = (line_perc.nlargest(10, 'percentage')).index

##per episode lines for main characters
line_main = line_sum.loc[line_sum['character'].isin(main_char)]

##Join main data with lines per character data

##convert line data into dataframe with each row as a season/episode and each column as the main character lines
char_lines = line_main.pivot_table(values='lines', index='season_ep', columns='character').reset_index()
char_lines = char_lines.fillna(0) ##fill NA values with zero because they are only not present if no lines were spoken

##calculate character line percentage per episode
perc_cols = char_lines.select_dtypes(include=np.number).columns + "_perc" ##create column names
char_lines[perc_cols] = char_lines.select_dtypes(include=np.number).div(d.n_lines, axis = 0) ##per row char line percentages

##copy original data so we don't lose it
df = d.copy()

##Note that some directors are repeated with different spellings. 
##Need to fix before any future data solutions
wrongdir_dict = {"Greg Daneils": "Greg Daniels", "Charles McDougal": "Charles McDougall",
                 "Claire Scanlong":"Claire Scanlon"}
df["director"].replace(wrongdir_dict, inplace=True)

##create season_ep column for d to join by
df['season_ep'] = df['season'].astype(str) + "_" + df['episode'].astype(str)

##join the main data with the lines per character data
df = pd.merge(df, char_lines, on = 'season_ep', how = 'left')


##Want to get a list of all writers and a list of all directors 
##Create new columns where the presence of each writer is it's own variable (one-hot encoding)

# convert string to lists
#df[["year", "month", "day"]] = pd.DataFrame(d.air_date.str.split("-").tolist()).astype(int)
df["writer"]                 = df["writer"].str.split(";")
df["main_chars"]             = df["main_chars"].str.split(";")
df["director"]               = df["director"].str.split(";")

##Now create dummy variables for each of the writers/directors (we are dropping main character dummies)

#######################################################################################################
##Functions to Modify Writer/Director##

##Function to split column with list variables into dummy variable columns
def split_col(column, df):
    ##For each episode, check which writiers or directosr are present and put into a dictionary
    ##Each column is a writer/direction; each row is an episode
    ##The value of row, column is T/F for whether a writer/director is present
    
    ##get all writers/directors in a flat list and then get unique writers/directors
    all_items = [item for ep_item in df[column] for item in ep_item] ##flattened list of writers/directors (duplicates)
    items = list(set(all_items)) ##gets list of unique writers/directors

    ##For each episode, check which writers/directors are present and put into a dictionary
    ##Each column is an individual writer/director; each row is an episode
    ##The value of row, column is T/F for whether a writer/director is present
    items_df = df[column].apply(lambda x: i in x for i in items)
    items_df = items_df.astype(int) ##convert boolean T/F to 1/0
    items_df.columns = [str(x) + "_dummy" + "_" + column for x in items] ##item names + "dummy"
    
    return(items_df)


##Function to group "low" appearance directors/writers into one "low-appearance" column dummy variable
##Removes the original columns for those low-appearane writers directors
def categorize_low(column_type, df, threshold):
    ##if writer/director shows up less than or equal to the threshold, re-categorize to "Low-Appearance"
    items_eps = df.sum(axis = 0)
    low_appearance = list(items_eps[items_eps <= threshold].index) 
    new_col = "low_appearance" + "_" + column_type

    ##create new writer column for low-appearance
    df[new_col] = df[low_appearance].sum(axis = 1) > 0 ##Identifies episodes where a low_appearance writer/dir is
    df[new_col] = df[new_col].astype(int) ##convert boolean to integer
    
    ##drop original low_appearance columns
    df.drop(columns = low_appearance, inplace = True)
    
    return(df)

##End of Functions##
##################################################################################################################

##create dummy variable dataframes for writers and directors
writers_df = split_col(column = "writer", df = df)
directors_df = split_col(column = "director", df = df)

##group low-appearance writers and directors and remove columns for those low-appearance writers/directors
writers_df = categorize_low(column_type = "writer", df = writers_df, threshold = 2)
directors_df = categorize_low(column_type = "directors", df = directors_df, threshold = 2)

##Add the writer data to overall data
df = pd.concat([df, writers_df, directors_df], axis = 1)


# Create column to indicate if said episode consist of multi parts
p = re.compile("Parts 1&2")
df["multi_part"] = [int(not int(pd.isnull(re.search(p,i)))) for i in df["episode_name"]]

##Replace any spaces in column names with "_"
df.columns = df.columns.str.replace(' ', '_')


#%%

# mlb = MultiLabelBinarizer()
# # Multiple OHE for columns with arrays
# ddat = pd.DataFrame(mlb.fit_transform(df.iloc[:,3]), columns=mlb.classes_+ "_dummy")
# wdat = pd.DataFrame(mlb.fit_transform(df.iloc[:,4]), columns=mlb.classes_+ "_dummy")
# adat = pd.DataFrame(mlb.fit_transform(df.iloc[:,12]), columns=mlb.classes_+ "_dummy")

# col_in = [i for i in range(0,df.shape[1])]
# col_in.remove(3)
# col_in.remove(4)
# col_in.remove(12)
# df   = pd.concat([df[df.columns[col_in]], ddat, wdat, adat], axis = 1)



# %%
df.to_csv("data/full_raw_dat.csv", index=False)

# remove unused columns and observations
col_drop = ["episode_name", "season_ep", "air_date", "episode"]
p        = re.compile("Part [12]")
row_drop = [pd.isnull(re.search(p,i)) for i in df["episode_name"]]
fdat     = df.drop(col_drop,axis=1).iloc[row_drop,:]

Path("data").mkdir(parents=True, exist_ok=True)
fdat.to_csv("data/full_filtered_dat.csv", index=False)