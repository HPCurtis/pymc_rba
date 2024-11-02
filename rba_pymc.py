import pandas as pd
import pymc as pm
import numpy as np
import bambi as bmb
import pytensor.tensor as pt
import time

# Import data
FILE_PATH = "https://raw.githubusercontent.com/HPCurtis/Datasets/refs/heads/main/rba.csv"
df = pd.read_csv(FILE_PATH)

# Covnert the strings to unique integers. 
df["roi_int"] = pd.factorize(df["ROI"])[0]
df["subject_int"] = pd.factorize(df["subject"])[0]

# Specify variables for pymc model.
y = df.y
N_ROI = len(np.unique(df.ROI))
N_subj = len(np.unique(df.subject))
ROI =  df.roi_int.values
subj = df.subject_int.values
Xc = (df.x.values - np.mean(df.x.values))
X = df.x.values
J = 2

# RBA model follwing the bambi syntax y ~ x + (1|subject) + (x|ROI) but with correlted slopes"
with pm.Model() as model:

    # Intercept (alpha) and beta (fixed) distribution
    alpha = pm.Normal("alpha", mu=0.17, sigma=4.66)
    beta = pm.Normal("beta", mu=0, sigma=0.27)
    sigma = pm.HalfStudentT("sigma",nu=4, sigma=2.3)

    # Subjct random intercept.
    tau_u = pm.HalfNormal("tau_u", sigma=4.66)
    z_u = pm.Normal('z_u', mu=0, sigma=1, shape=N_subj)
    u = pm.Deterministic("u", z_u * tau_u)

    # Randomintecept and slope correlated.
    z_u2 = pm.Normal('z_u2', 0., 1., shape=(J, N_ROI))
      
    sd_dist = pm.HalfNormal.dist(sigma=[4.66,0.27], shape=J)
    L_u, rho, tau_u2 = pm.LKJCholeskyCov('L_u',
                                        eta=1, n=J,
                                        sd_dist=sd_dist)
    u2 = pm.Deterministic("u2", pt.dot(L_u, z_u2).T)

    # Likelihood
    mu = alpha + beta * Xc + u[subj] + u2[ROI, 0] + X * u2[ROI, 1]
    y = pm.Normal('y', mu = mu, sigma = sigma, observed=y)
    
fig = pm.model_to_graphviz(model)
fig.render("model_graph_pymx", format= "png")

# Time model fitting.
start_time = time.time()

with model:
    # Fit nutpie model for fastest cpu performance.
    pm.sample(nuts_sampler="nutpie", draws=1000, tune=1000, 
              chains=4, cores=8, target_accept=0.8)

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
