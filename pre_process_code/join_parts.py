

import os
import pandas as pd
import argparse


list_dfs = os.listdir('../splitted/primevul')

big_df = pd.DataFrame()
for dataframe in list_dfs:
    df_new = pd.read_csv('./splitted/primevul/'+ dataframe)
    if len(big_df) == 0:
        big_df = df_new
    else:
        big_df = pd.concat([big_df,df_new])


big_df.to_csv('/home/oso/Documents/incibe/code/pre_processing/multimetric/tests/processed/joined_primevul_metadata2.csv')