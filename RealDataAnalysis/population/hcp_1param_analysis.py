##conda activate fmri_inv_rda
import numpy as np 
import math
import torch 
import torch.nn as nn
from torch.special import gammaln, xlogy
from torch.utils.data import DataLoader, Dataset

import scipy

import trimesh
import igl
import scipy
from sksparse.cholmod import cholesky
import nibabel as nib 

import pandas as pd 
import statsmodels.formula.api as smf 

import os 
from scipy.io import loadmat 
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.colors import Normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### Forward Model HRF Kernel ####
## constants
DT = 0.001
P_DELAY = 6.0
UNDERSHOOT = 16.0
P_DISP = 1.0
U_DISP = 1.0
P_U_RATIO = 0.167

## double gamma HRF model
A_1 =  torch.tensor([P_DELAY / P_DISP - 1])
A_2 = torch.tensor([UNDERSHOOT / U_DISP - 1])
C = torch.tensor([P_U_RATIO])

def inv_link(theta_tilde, min_theta, max_theta):
	u = torch.distributions.Normal(0, 1).cdf(torch.from_numpy(theta_tilde).float()).cpu().detach().numpy()
	return min_theta + (max_theta - min_theta) * u

def double_gamma_hrf(theta, t):
	def _gamma_pdf(t, a, b):
		## General gamma PDF: (b^(a+1) * t^a * exp(-b*t)) / Gamma(a+1)
		eps = 1e-8  # avoid t=0 instability
		t = torch.clamp(t, min=eps)
		norm = torch.exp(gammaln(a + 1))
		return (b ** (a + 1)) * (t**a) * torch.exp(-b * t) / norm
	peak = _gamma_pdf(t, A_1, theta)
	undershoot = _gamma_pdf(t, A_2, theta)
	hrf = peak - C * undershoot
	return hrf, t

def time_to_peak_from_model(theta, tgrid):
    hrf, t = double_gamma_hrf(theta, tgrid) 
    if hrf.ndim == 1:
        idx = torch.argmax(hrf).item()
        return t[idx].item()
    else:
        idx = torch.argmax(hrf, dim=1)           # (batch,)
        return t[idx].cpu().numpy()              # seconds

#### Load Functions ####
subj_ids = np.unique([fname.split(".")[0] for fname in os.listdir("S1200_Rest3TRecommended")])
N = len(subj_ids)
V = 32492

theta_tilde = np.zeros((N, V))
for i, subj_id in enumerate(subj_ids):
	DATA_FILE = "S1200_RestT3_Estimates/%s_1param_estims.npy" % (subj_id,)
	data_id = np.load(DATA_FILE, allow_pickle=True).item()
	theta_tilde_hat_id = data_id["theta_tilde_hat"]
	theta_tilde[i,:] = theta_tilde_hat_id.ravel()
	print("Finished", DATA_FILE)

#### fPCA ####
## get template domain
MESH_FILE = "S1200_templates/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
mesh = nib.load(MESH_FILE)
verts = mesh.darrays[0].data 
tris = mesh.darrays[1].data

## build inner product matrix
C = igl.massmatrix(verts, tris) 

## get forward model 
t1_min, t1_max = 0.5, 2.5
eps_1 = 0.01 / 0.98 * (t1_max - t1_min)

if not os.path.exists("fpca/fpca.npz"):
	## generalized eigen-decomp 
	Cov_theta = (1./N) * C @ (theta_tilde.T @ theta_tilde) @ C
	eigvals, eigvecs = scipy.linalg.eigh(Cov_theta, C.todense())
	idx = np.argsort(eigvals)[::-1]
	eigvals = eigvals[idx]
	eigvecs = eigvecs[:,idx]
	np.savez("fpca/fpca.npz", {"coordinates":verts, 
						"faces":tris, 
						"eigen_funcs":eigvecs.astype(np.float32),
						"eigen_vals":eigvals.astype(np.float32)})
	mu_theta_tilde = theta_tilde.mean(axis=0)
	np.savez("fpca/mu.npz", {"mu_theta_tilde":mu_theta_tilde})
	mu_theta = inv_link(mu_theta_tilde, t1_min - eps_1, t1_max + eps_1)
	tgrid = torch.linspace(0,30,1000)
	mu_TTP = time_to_peak_from_model(torch.from_numpy(mu_theta).float().view(-1,1), tgrid)
	np.savez("fpca/mu_TTP.npz", {"mu_TTP":mu_TTP})
else:
	data = np.load("fpca/fpca.npz", allow_pickle=True)["arr_0"].item()
	eigvecs = data["eigen_funcs"]
	eigvals = data["eigen_vals"]
	data_mu = np.load("fpca/mu.npz", allow_pickle=True)["arr_0"].item()
	mu_theta_tilde = data_mu["mu_theta_tilde"]


## spectral rank 
K = 10
eigvals = eigvals[:K]
eigvecs = eigvecs[:,:K]

theta_tilde_proj = theta_tilde @ eigvecs

#### Inference ####

#### Load Covariates ####
df_covar = pd.read_csv("SMs.csv")
df_conf = pd.read_csv("conf.csv")

ids = list(map(int, subj_ids))
order_df = pd.DataFrame({"subject": ids})

df_covar = order_df.merge(df_covar, on="subject", how="inner")
df_conf = order_df.merge(df_conf, on="subject", how="inner")

data = pd.DataFrame(theta_tilde_proj, columns=["PC%s"%(k+1) for k in range(K)])
data["subject"] =  ids

data = data.merge(df_conf, on="subject", how="inner")
data = data.merge(df_covar, on="subject", how="inner")
data["Sex"] = (data["Sex"] > 0).astype(int)  
data["HeavyDrinker"] = data.apply(lambda row: 1 if row.SSAGA_Alc_Hvy_Drinks_Per_Day >= 5 else 0, 
						axis=1) 

logreg_sex = smf.logit("Sex ~ PC1 + PC2 + PC3 + PC4 + PC5", data=data).fit()
print(logreg_sex.summary())          
	   
lm_cigs = smf.ols("Times_Used_Any_Tobacco_Today ~ PC1 + PC2 + PC3 + PC4 + PC5", data=data).fit()
print(lm_cigs.summary())   

logreg_mj = smf.logit("SSAGA_Mj_Ab_Dep ~ PC1 + PC2 + PC3 + PC4 + PC5", data=data).fit()   
print(logreg_mj.summary())   



