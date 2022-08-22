import pandas as pd
import numpy as np

def train_validate_test_split(df, percent=.7, seed=None):
    df = df.reset_index()
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(percent * m)
    
    train_df = df.iloc[perm[:train_end]]
    eval_df = df.iloc[perm[train_end:]]
    return train_df, eval_df
