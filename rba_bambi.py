import pandas as pd
import pymc as pm
import numpy as np
import bambi as bmb
import bayeux as bx
import time

# Import data
FILE_PATH = "https://raw.githubusercontent.com/HPCurtis/Datasets/refs/heads/main/rba.csv"
df = pd.read_csv(FILE_PATH)

# Time model fitting.
start_time = time.time()

# fit bambi model
model = bmb.Model("y ~ x + (1|subject) + (x|ROI)", data=df)
model.build()
fitted = model.fit(tune=1000, 
                   draws=1000, 
                   chains=4, 
                   inference_method="numpyro_nuts")
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")