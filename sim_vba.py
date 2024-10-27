import pandas as pd
import pymc as pm
import numpy as np

FILE_PATH = "df2.csv"
df = pd.read_csv(FILE_PATH)

y = df.BOLD
N_voxel = len(np.unique(df.voxel_id))
N_subj = len(np.unique(df.participant))
voxel =  df.voxel_id.values
subj = df.participant.values
#Xc = (df.x - np.mean(df.x)) / np.std(df.x)
X = df.condition.values

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
    tau_u2 = pm.HalfStudentT("tau_u2", nu=3, sigma=2.5)
    z_u2 = pm.Normal("z_u2", 0, 1, shape=N_voxel)
    u2 = z_u2 * tau_u2

    # Likelihood
    mu = alpha + beta * X + u[subj] + u2[voxel] 
    y = pm.Normal('y', mu = mu, sigma = sigma ,observed=y)

with model:
    pm.sample(nuts_sampler="nutpie",
                      draws=1000, tune=1000, 
                      chains=4, cores=8, target_accept = .8)