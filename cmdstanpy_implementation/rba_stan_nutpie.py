# Import relevant data analysis and visualisation packages.
import numpy as np
import pandas as pd
from patsy import dmatrix
import nutpie

MOD_FILE_PATH = "stanfiles/rba.stan"
FILE_PATH = "https://raw.githubusercontent.com/HPCurtis/Datasets/refs/heads/main/rba.csv"
df = pd.read_csv(FILE_PATH)

# Stan is 1 based indexing so additon of one required.
df["int_roi"] = pd.factorize(df["ROI"])[0] + 1
df["int_subj"] = pd.factorize(df["subject"])[0] +1 

#Genrate dseign matrix
X = np.asarray(dmatrix("x", data = df) )
"""
data = {
    'N': df.shape[0],  # number of rows in df
    'y': df['y'].values,  # assuming 'y' is a column in df
    'X': X,  # assuming X is already defined as a numpy array or similar
    'K': X.shape[1],  # number of columns in X
    'Kc': X.shape[1] - 1,  # K minus 1
    'J': 2,  # fixed value
    'N_subj': len(df['subject'].unique()),  # number of unique subjects
    'N_ROI': len(df['ROI'].unique()),  # number of unique ROIs
    'subj': df['int_subj'].values,  # assuming 'int_subj' is a column in df
    'ROI': df['int_roi'].values,  # assuming 'int_roi' is a column in df
    #'grainsize': round(df.shape[0] / 4)  # grainsize calculation
}
# Creat json file of the data.
#write_stan_json("data.json", data = data)
"""

compiled = nutpie.compile_stan_model(filename="stanfiles/rba.stan")

# Provide data
compiled = compiled.with_data(
    N=df.shape[0],
    y=df['y'].values.tolist(),
    X=X.tolist(),
    K=X.shape[1],
    Kc=X.shape[1] - 1,
    J=2,
    N_subj=len(df['subject'].unique()),
    N_ROI=len(df['ROI'].unique()),
    subj=df['int_subj'].values.tolist(),
    ROI=df['int_roi'].values.tolist()
)
fit = nutpie.sample(compiled, draws=1000, tune=1000)