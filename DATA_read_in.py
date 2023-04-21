#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 10:27:25 2023

@author: horvathbalint
"""
from tqdm import tqdm
import glob
import pyreadstat
import numpy as np
import pandas as pd
from typing import List
import os
#import threading
import pyarrow
#from concurrent.futures import ProcessPoolExecutor

#working directory
#os.chdir('/Volumes/T7/OPTION_PRICE')

def read_parquet_files(file_list: List[str]) -> pd.DataFrame:
    """
    Reads a list of Parquet files into a single Pandas dataframe.
    """
    dfs = []
    for f in file_list:
        df = pd.read_parquet(f)
        dfs.append(df)
    return pd.concat(dfs)

#file path
sasfile = '/Users/horvathbalint/Data/Options/Raw_SAS/security_price.sas7bdat'

if __name__ == '__main__': 
    reader = pyreadstat.read_file_in_chunks(pyreadstat.read_sas7bdat, sasfile, chunksize=1e7, multiprocess=True, num_processes=50)
    for df, meta in tqdm(reader):
        df_columns = df[df['SecurityID'].isin([501271.0, 506496.0, 508037.0, 506522.0, 510399.0, 506528.0, 506552.0, 707745.0])]
        r = np.random.randint(0, 1000000000)
        df_columns.to_parquet(f"{sasfile.replace('.sas7bdat', '')}_{r}.parquet", engine='pyarrow')
#    pq_files = glob.glob(f"{sasfile.replace('.sas7bdat', '')}*.parquet")
#    df_columns = read_parquet_files(pq_files)
#    df_columns.to_parquet(sasfile.replace("sas7bdat", "parquet"))
     
#    for f in pq_files:
#        os.remove(f)
        




