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
    z_u = pm.Normal('z_u', mu=0, sigma=1, shape=N_subj)
    u = pm.Deterministic("u", z_u * tau_u)

    # Randomintecept and slope correlated.
    z_u2 = pm.Normal('z_u2', 0., 1., shape=(J, N_ROI))
      
    sd_dist = pm.HalfNormal.dist(sigma=[4.66,0.27], shape=J)
    L_u, rho, tau_sd = pm.LKJCholeskyCov('L_u',
                                        eta=1, n=J,
                                        sd_dist=sd_dist)
    
    rho = pm.Deterministic("rho", rho[0, 1])

    tau_u2 = pm.Deterministic("tau_u2", tau_sd)
    u2 = pm.Deterministic("u2", pt.dot(L_u, z_u2).T, dims=("ROI", "effect"))

    # Likelihood
    mu = alpha + beta * Xc + u[subjects_id] + u2[ROI_id, 0] + X * u2[ROI_id, 1]
    y = pm.Normal('y', mu = mu, sigma = sigma, observed=y)
    
if not os.path.exists("vis/model_graph_pymc.png"):
    fig = pm.model_to_graphviz(model)
    fig.render("model_graph_pymc", format="png")
    
# Time model fitting.
start_time = time.time()

with model:
    # Fit Numpyro sampler for correlation model for fastest cpu performance.
    fit = pm.sample(nuts_sampler="numpyro", draws=1000, tune=1000, 
              chains=4, cores=4, target_accept=0.9)

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")

# Sample from the posterior predictive distribution.
with model:
    pm.sample_posterior_predictive(fit, extend_inferencedata=True)

print(az.summary(fit, var_names = ["alpha", "beta", "sigma", "tau_u", "tau_u2","rho"]  ))

# Plot MCMC trace. 
az.plot_trace(fit, var_names = ["alpha", "beta", "sigma", "tau_u", "tau_u2", "rho"])
# Rank plot of MCMC chains.
az.plot_rank(fit, var_names = ["alpha", "beta", "sigma", "tau_u", "tau_u2", "rho"])
# Fores plot.
az.plot_forest(fit, var_names="u2", combined= True, hdi_prob=.95);
plt.show()