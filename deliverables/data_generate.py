#%%
# Data libraries
import pandas as pd
import numpy as np
import re

# sklearn modules
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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
char_lines[char_lines.select_dtypes(include=np.number).columns + "_perc"] = char_lines.select_dtypes(include=np.number).div(d.n_lines, axis = 0)
#%%
df = d.copy()

# convert string to lists
#df[["year", "month", "day"]] = pd.DataFrame(d.air_date.str.split("-").tolist()).astype(int)
df["writer"]                 = df["writer"].str.split(";")
df["main_chars"]             = df["main_chars"].str.split(";")
df["director"]               = df["director"].str.split(";")

##create season_ep column for d to join by
df['season_ep'] = df['season'].astype(str) + "_" + df['episode'].astype(str)

##join the main data with the lines per character data
df = pd.merge(df, char_lines, on = 'season_ep', how = 'left')

# Create column to indicate if said episode consist of multi parts
p = re.compile("Parts 1&2")
df["multi_part"] = [int(not int(pd.isnull(re.search(p,i)))) for i in df["episode_name"]]
#%%

mlb = MultiLabelBinarizer()
# Multiple OHE for columns with arrays
ddat = pd.DataFrame(mlb.fit_transform(df.iloc[:,3]), columns = mlb.classes_+ "_dummy")
wdat = pd.DataFrame(mlb.fit_transform(df.iloc[:,4]), columns = mlb.classes_+ "_dummy")
adat = pd.DataFrame(mlb.fit_transform(df.iloc[:,12]), columns = mlb.classes_+ "_dummy")

col_in = [i for i in range(0,df.shape[1])]
col_in.remove(3)
col_in.remove(4)
col_in.remove(12)
df   = pd.concat([df[df.columns[col_in]], ddat, wdat, adat], axis = 1)



# %%
df.to_csv("full_raw_dat.csv")

# remove unused columns and observations
col_drop = ["episode_name", "season_ep", "air_date", "episode"]
p        = re.compile("Part [12]")
row_drop = [pd.isnull(re.search(p,i)) for i in df["episode_name"]]
fdat     = df.drop(col_drop,axis=1).iloc[row_drop,:]

# convert season to object
fdat.season = fdat.season.astype(str)

fdat.to_csv("full_filtered_dat.csv")
# %%

X = fdat.drop("imdb_rating", axis=1)
y = fdat.imdb_rating

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=5)
X_train.to_csv("split_X_train_raw.csv")
X_test.to_csv("split_X_test_raw.csv")
y_train.to_csv("split_y_train_raw.csv")
y_test.to_csv("split_y_test_raw.csv")

