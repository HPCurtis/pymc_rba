import pandas as pd
import pymc as pm
import numpy as np
import bambi as bmb
import bayeux as bx
import time
import os

# Import data
FILE_PATH = "https://raw.githubusercontent.com/HPCurtis/Datasets/refs/heads/main/rba.csv"
df = pd.read_csv(FILE_PATH)

# Time model fitting.
start_time = time.time()

# fit bambi model and time it.
model = bmb.Model("y ~ x + (1|subject) + (x|ROI)", data=df)
model.build()
fitted = model.fit(chains=4,
                   draws=1000,
                   tune=1000, 
                   inference_method="nutpie")
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")

if not os.path.exists("model_bambi_graph.png"):
    fig = model.graph()
    fig.render(filename="model_bambi_graph", format="png")

print(model)