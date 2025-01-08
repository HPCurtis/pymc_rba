import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import bambi as bmb
import pytensor.tensor as pt
import arviz as az
import time
import os

# Import data
FILE_PATH = "https://raw.githubusercontent.com/HPCurtis/Datasets/refs/heads/main/rba.csv"
df = pd.read_csv(FILE_PATH)

subject, subjects_id = np.unique(df["subject"], return_inverse=True)
ROI, ROI_id = np.unique(df["ROI"], return_inverse=True)

# Specify variables for pymc model.
y = df.y
N_ROI = len(np.unique(df.ROI))
N_subj = len(np.unique(df.subject))

Xc = (df.x.values - np.mean(df.x.values))
X = df.x.values
J = 2

coords = {"ROI": ROI, "subject": subject, "effect": ["intercept", "slope"]}
# RBA model follwing the bambi syntax y ~ x + (1|subject) + (x|ROI) but with correlated slopes"
with pm.Model(coords=coords) as model:

    # Intercept (alpha) and beta (fixed) distribution
    alpha = pm.Normal("alpha", mu=0.17, sigma=4.66)
    beta = pm.Normal("beta", mu=0, sigma=0.27)
    sigma = pm.HalfStudentT("sigma",nu=4, sigma=2.3)

    # Subjct random intercept.
    tau_u = pm.HalfNormal("tau_u", sigma=4.66)
    z_u = pm.Normal('z_u', mu=0, sigma=1, dims="subject")
    u = pm.Deterministic("u", z_u * tau_u)

    # Randomintecept and slope correlated.
    z_u2 = pm.Normal('z_u2', 0., 1., shape=(N_ROI, J))
      
    tau_u2 = pm.HalfNormal("tau_u2", sigma=4.66, shape= J)
    u2_int = pm.Deterministic("u2_int", z_u2[:,0] * tau_u2[0], dims="ROI")
    u2_slope = pm.Deterministic("u2_slope", z_u2[:,1] * tau_u2[1], dims="ROI")

    # Likelihood
    mu = alpha + beta * Xc + u[subjects_id] + u2_int[ROI_id] + X * u2_slope[ROI_id]
    y = pm.Normal('y', mu = mu, sigma = sigma, observed=y)
    
if not os.path.exists("vis/model_graph_pymc_uncor.png"):
    fig = pm.model_to_graphviz(model)
    fig.render("model_graph_pymc_uncor", format="png")
    
# Time model fitting.
start_time = time.time()

with model:
    # Fit nutpie model for fastest cpu performance without correlation.
    fit = pm.sample(nuts_sampler="nutpie", draws=1000, tune=2000, 
              chains=4, cores=4, target_accept=0.95)

end_time = time.time()
 
print(f"Execution time: {end_time - start_time} seconds")

print(az.summary(fit, var_names = ["alpha", "beta", "sigma", "tau_u", "tau_u2"] ))

az.plot_trace(fit, var_names = ["alpha", "beta", "sigma", "tau_u", "tau_u2"])
az.plot_rank(fit, var_names = ["alpha", "beta", "sigma", "tau_u", "tau_u2"])
az.plot_forest(fit, var_names="u2_slope", combined= True, hdi_prob=.95)
plt.show()