import pandas as pd
import pymc as pm
import numpy as np
import bambi as bmb

FILE_PATH = "https://raw.githubusercontent.com/HPCurtis/Datasets/refs/heads/main/rba.csv"
df = pd.read_csv(FILE_PATH)

df["roi_int"] = pd.factorize(df["ROI"])[0]
df["subject_int"] = pd.factorize(df["subject"])[0]

y = df.y
N_ROI = len(np.unique(df.ROI))
N_subj = len(np.unique(df.subject))
ROI =  df.roi_int.values
subj = df.subject_int.values
Xc = (df.x - np.mean(df.x)) / np.std(df.x)
X = df.x.values
J = 2

"""
# Horrible slow on cpu take huge time compared to my implementation.
model = bmb.Model("y ~ x + (1|subject) + (x|ROI)", data=df)
fitted = model.fit(tune=1000, 
                   draws=1000, 
                   chains=4, 
                   method='numpyro_nuts',
                   nuts_kwargs=dict(max_tree_depth=100)
                   )

"""
with pm.Model() as model:

    # Intercept (alpha) and beta (fixed) distribution
    alpha = pm.StudentT("alpha", nu = 3, mu=0, sigma=2.5)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfStudentT("sigma",nu=3, sigma=2.5)

    # Subjct random intercept.
    tau_u = pm.HalfStudentT("tau_u", nu=3, sigma=2.5)
    z_u = pm.Normal('z_u', mu=0, sigma=1, shape=N_subj)
    u = z_u * tau_u

    # Uncentered parameterization for each variable
    tau_u2 = pm.HalfStudentT("tau_u2", nu=3, sigma=2.5, shape=J)
    z_u2 = pm.Normal("z_u2", 0, 1, shape=(J, N_ROI))
    u2_1 = z_u2[0] * tau_u2[0]
    u2_2 = z_u2[1] * tau_u2[1]

    # Likelihood
    mu = alpha + beta * Xc + u[subj] + u2_1[ROI] + X * u2_2[ROI]
    y = pm.Normal('y', mu = mu, sigma = sigma ,observed=y)

with model:
    pm.sample(nuts_sampler="nutpie",
                      draws=1000, tune=1000, 
                      chains=4, cores=8, target_accept = .8)
