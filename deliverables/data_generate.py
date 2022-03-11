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
main_char = line_perc.loc[line_perc['percentage'] > 0.005].index

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
ddat = pd.DataFrame(mlb.fit_transform(df.iloc[:,3]), columns=mlb.classes_+ "_dummy")
wdat = pd.DataFrame(mlb.fit_transform(df.iloc[:,4]), columns=mlb.classes_+ "_dummy")
#adat = pd.DataFrame(mlb.fit_transform(df.iloc[:,12]), columns=mlb.classes_+ "_dummy")

col_in = [i for i in range(0,df.shape[1])]
col_in.remove(3)
col_in.remove(4)
col_in.remove(12)
df   = pd.concat([df[df.columns[col_in]], ddat, wdat], axis = 1)



# %%
df.to_csv("data/full_raw_dat.csv", index=False)

# remove unused columns and observations
col_drop = ["episode_name", "season_ep", "air_date", "episode", "total_votes", "season", "writer", "director"]
p        = re.compile("Part [12]")
row_drop = [pd.isnull(re.search(p,i)) for i in df["episode_name"]]
fdat     = df.drop(col_drop,axis=1).iloc[row_drop,:]

Path("data").mkdir(parents=True, exist_ok=True)
fdat.to_csv("data/full_filtered_dat.csv", index=False)
# %%

import rpy2
from rpy2 import robjects


#%%
label_dat = pd.DataFrame({
"labels" : ['S1','S2','S3','S4','S5','S6','S7','S8','S9'],
"season_ep" : ['S1-EP3','S2-EP11','S3-EP12','S4-EP11','S5-EP14','S6-EP13','S7-EP13','S8-EP12','S9-EP11']})

vline_mark = pd.DataFrame({
"episode" : ['7','29','52','66','92','116','140','164','187'],
"season" : ['1','2','3','4','5','6','7','8','9']})

season_brks = ['S1-EP1','S1-EP6','S2-EP5','S2-EP10','S2-EP15','S2-EP20','S3-EP3','S3-EP8','S3-EP14','S3-EP19','S3-EP24','S4-EP9','S4-EP14','S5-EP1','S5-EP7','S5-EP12','S5-EP18','S5-EP23','S5-EP28','S6-EP6','S6-EP11','S6-EP16','S6-EP22','S7-EP1','S7-EP6','S7-EP11','S7-EP17','S7-EP22','S8-EP2','S8-EP7','S8-EP12','S8-EP17','S8-EP22','S9-EP3','S9-EP8','S9-EP13','S9-EP18','S9-EP24']
# %%
df.season_ep = d.apply(lambda x : "S" + str(x["season"]) + "-EP" + "" + str(x["episode"]), axis = 1)
# %%
df = df.merge(label_dat, how = "left", on="season_ep")
# %%
import plotnine as pn 
df.season_ep = pd.Categorical(df.season_ep, categories=df.season_ep, ordered=True
)

# %%
p = (pn.ggplot(df) +
pn.aes(x = "season_ep", y = "imdb_rating") +
pn.geom_point(shape = 0)  +
pn.geom_line(pn.aes(group = 1)) +
pn.geom_vline(xintercept = vline_mark.episode, linetype = 'dashed') +
pn.geom_text(pn.aes(y = 10, label = "labels"), colour = 'black', size = 8) +
#pn.scale_x_continuous(breaks = [1], labels = "season_ep") +
pn.scale_x_discrete(breaks = season_brks) +
pn.ylim([6.5,10]) +
pn.theme_minimal()+
pn.theme(axis_text_x = pn.element_text(angle = -45, hjust = -.05)))

# %%
