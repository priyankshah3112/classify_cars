from copy import deepcopy
import os
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)
from config import config

# cleaning the column names in raw file by removing spaces and replacing it by "_"
def clean_column_names(df_original):

    df=deepcopy(df_original)
    for col in df_original.columns.values:
        new_col=col.replace(" ","_")
        df.rename(columns={col:new_col},inplace=True)

    return df

# converting string columns to numeric data using the feature_mapper defined in config file
def make_columns_numeric(df_original):

    df=deepcopy(df_original)
    feature_mapper=config.feature_mapper

    for col in df.columns.values:
        if col in feature_mapper:
            df[col]=df[col].map(feature_mapper[col])

    return df