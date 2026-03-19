import torch 
import torch.nn as nn
from torch.special import gammaln, xlogy
from torch.utils.data import DataLoader, Dataset

import scipy
from scipy.fft import fft 

import trimesh
import igl

import numpy as np 

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import fdrcorrection

import pandas as pd 

import matplotlib.pyplot as plt 

torch.manual_seed(0)
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

class HRF(nn.Module):
	def __init__(self, t_r=1.0, T=100, L=30, onset=0.0):
		super().__init__()
		self.T = T
		self.t_r = t_r 
		self.dur = L * t_r
		self.onset = onset
		self.L = L
	def _gamma_pdf(self, t, a, b):
		## General gamma PDF: (b^(a+1) * t^a * exp(-b*t)) / Gamma(a+1)
		eps = 1e-8  # avoid t=0 instability
		t = torch.clamp(t, min=eps)
		norm = torch.exp(gammaln(a + 1))
		return (b ** (a + 1)) * (t**a) * torch.exp(-b * t) / norm
	def _double_gamma_hrf(self, theta):
		t = torch.linspace(0, self.dur, int(float(self.dur) / DT), device=theta.device, dtype=theta.dtype) - float(self.onset) / DT
		t = t[::int(self.t_r/DT)]
		peak = self._gamma_pdf(t, A_1, theta)
		undershoot = self._gamma_pdf(t, A_2, theta)
		hrf = peak - C * undershoot
		return hrf, t
	def _construct_toeplitz(self, kernel):
		"""
			Construct the Toeplitz matrix for a 1D convolution kernel.

			Args:
				kernel (torch.Tensor): The 1D convolution kernel of shape (L,).
			Returns:
				torch.Tensor: The Toeplitz matrix of shape (signal_length, signal_length).
		"""
		toeplitz_matrix = torch.zeros((self.T, self.T + self.L - 1))
		for i in range(self.T):
			toeplitz_matrix[i, i:i + self.L] = kernel
		return toeplitz_matrix.T[:self.T,:]
	def forward(self, theta):
		hrf_kernels, _ = self._double_gamma_hrf(theta)
		H = torch.stack([self._construct_toeplitz(k) for k in hrf_kernels])
		return H  

def wienerDeconv(y_x, h_x):
	## fft of HRF and observed signal
	h_omega_x = torch.fft.rfft(h_x,  n=M+L-1)
	y_omega_x = torch.fft.rfft(y_x, n=M+L-1)
	## calculate Weiner deconv in frequency space 
	s_omega_x_hat = (torch.conj(h_omega_x) * y_omega_x) / (h_omega_x.abs() ** 2 + lambda_reg)
	## transform to time domain 
	s_x_hat = torch.fft.irfft(s_omega_x_hat, n=M+L-1)[:M]
	return s_x_hat.cpu().detach().numpy()

def compute_gc_seed_vs_rest(X, labels, seed_ROI, maxlag=2, metric="neglogp", skip_seed=True):
	## Step 0: type and value checks 
	X = np.asarray(X)
	labels = np.asarray(labels)
	if X.ndim != 2:
		raise ValueError(f"X must be 2D (T x V), got shape {X.shape}")
	if labels.ndim != 1:
		raise ValueError(f"labels must be 1D, got shape {labels.shape}")
	T, V = X.shape
	if labels.shape[0] != V:
		raise ValueError("labels length must match number of vertices in X")
	# Seed parcel mask and average time series
	seed_mask = (labels == seed_ROI)
	if not seed_mask.any():
		raise ValueError(f"No vertices found with label {seed_ROI}")
	## Step 1: get average time series in the seed
	seed_ts = X[:, seed_mask].mean(axis=1)  # (T,)
	## granger causility seed -> vertex 
	## granger causlity vertex -> seed 
	gc_seed_to_vert = np.full(V, np.nan, dtype=float)
	gc_vert_to_seed = np.full(V, np.nan, dtype=float)
	for v in range(V):
		if skip_seed and seed_mask[v]:
			# do not look at GC between vertices within the same ROI 
			continue
		## get time series at mesh vertex v 
		target_ts = X[:, v]
		## do not analyze if time series is all-constant or NaN data (e.g. in the medial wall)
		if not np.isfinite(target_ts).all():
			continue
		if np.std(target_ts) == 0 or np.std(seed_ts) == 0:
			continue
		## Run GC for seed -> target: x = [target, seed], test "second column causes first"
		pair1 = np.column_stack([target_ts, seed_ts])
		try:
			res1 = grangercausalitytests(pair1, maxlag=maxlag, verbose=False)
		except Exception:
			# if statsmodels chokes (e.g., singular matrix), skip
			continue
		# get results 
		F1, p1, *_ = res1[maxlag][0]["ssr_ftest"]
		## Run GC fo target -> seed: x = [seed, target], test "second column causes first"
		pair2 = np.column_stack([seed_ts, target_ts])
		try:
			res2 = grangercausalitytests(pair2, maxlag=maxlag, verbose=False)
		except Exception:
			continue
		# get results 
		F2, p2, *_ = res2[maxlag][0]["ssr_ftest"]
		# Decide what to store
		if metric == "neglogp":
			gc_seed_to_vert[v] = -np.log10(p1 + 1e-12)
			gc_vert_to_seed[v] = -np.log10(p2 + 1e-12)
		elif metric == "F":
			gc_seed_to_vert[v] = F1
			gc_vert_to_seed[v] = F2
		elif metric == "p":
			gc_seed_to_vert[v] = p1
			gc_vert_to_seed[v] = p2
		else:
			raise ValueError(f"Unknown metric '{metric}'")
	return gc_seed_to_vert, gc_vert_to_seed

def proportion_sig_by_roi(labels, reject_mask, seed_ROI=None):
	prop_sig = {}
	for roi in np.unique(labels):
		if seed_ROI is not None and roi == seed_ROI:
			continue  # skip seed ROI if desired
		roi_mask = (labels == roi)
		n_vertices = roi_mask.sum()
		if n_vertices == 0:
			continue
		# Proportion of significant discoveries in this ROI
		prop_sig[roi] = reject_mask[roi_mask].mean()  # mean of {0,1} == proportion
	prop_sig_list = sorted(prop_sig.items(), key=lambda kv: kv[1], reverse=True)
	prop_sig_list = [(atlas_roi_name_dict.get(e[0]-180), e[1]) for e in prop_sig_list]
	return prop_sig_list

HRF_FILE = "hemo_inf/103818_inference_1param.npy"
SIGNAL_FILE = "data/103818_1_LR.L.1D"
MESH_FILE = "data/103818.L.midthickness.32k_fs_LR.ply" 
## Step 1: Load observed data 

y_obs = np.loadtxt(SIGNAL_FILE); TR = 0.72
#t = np.array([i*TR for i in range(y_obs.shape[1])])
mesh_verts, M = y_obs.shape

## get mask (caused by mesh hole)
filter_vertices = y_obs.std(axis=-1) == 0
cc_mask = (~filter_vertices)        
obs_idx = np.nonzero(cc_mask)[0]  
V = len(obs_idx)

## get surface mesh
mesh = trimesh.load(MESH_FILE)
verts = np.array(mesh.vertices)
tris = np.array(mesh.faces)

## Step 2: Load inferred hemodynamic field 
## HCP-parameters 
t_r =  TR
L = 30
onset = 0.0

## HCP-calibration 
t1_min, t1_max = 0.5, 2.5
gamma_hat = {'sigma_e': 0.15150529651120925,
			 'amin': 0.7435368830394751,
			 'amax': 0.8372886914947318,
			 'lam_min': 0.0039519025340883215,
			 'lam_max': 0.2167510244193856
			 }

sigma_e = gamma_hat["sigma_e"]
amin, amax = gamma_hat["amin"], gamma_hat["amax"]
lam_min, lam_max = gamma_hat["lam_min"], gamma_hat["lam_max"]

hrf_model = HRF(t_r=t_r, T=M, L=L, onset=onset)   

data = np.load(HRF_FILE,allow_pickle=True).item()
coordinates = data["coordinates"]
faces = data["faces"]
theta_hat = torch.from_numpy(data["theta_hat"]).float()

## Step 3: Invert signals per voxel VIA weiner deconvolution
hrf_kernel_theta_null, _ = hrf_model._double_gamma_hrf(torch.tensor([1]).float())
lambda_reg = sigma_e ** 2
shats = np.zeros((mesh_verts,M))
shats_null = np.zeros((mesh_verts,M))
for v in range(mesh_verts):
	if cc_mask[v]:
		## get local kernel and signals 
		hrf_kernel_theta_v, _ = hrf_model._double_gamma_hrf(theta_hat[v])
		y_obs_v = (y_obs[v,:] - y_obs[v,:].mean())/y_obs[v,:].std()
		## WD canoncial HRF 
		s_v_hat_0 = wienerDeconv(torch.from_numpy(y_obs_v).float(), hrf_kernel_theta_null)
		## WD spatially varying HRF 
		s_v_hat = wienerDeconv(torch.from_numpy(y_obs_v).float(), hrf_kernel_theta_v)
		shats[v,:] = s_v_hat
		shats_null[v,:] = s_v_hat_0


y_obs_standardized = (y_obs[obs_idx,:] - y_obs[obs_idx,:].mean(axis=-1).reshape(-1,1))/(y_obs[obs_idx,:].std(axis=-1).reshape(-1,1))
shats_null_standardized = (shats_null[obs_idx,:] - shats_null[obs_idx,:].mean(axis=-1).reshape(-1,1))/(shats_null[obs_idx,:].std(axis=-1).reshape(-1,1))
shats_standardized = (shats[obs_idx,:] - shats[obs_idx,:].mean(axis=-1).reshape(-1,1))/(shats[obs_idx,:].std(axis=-1).reshape(-1,1))

y_obs_standardized_full = np.zeros((mesh_verts, M))
shats_null_standardized_full = np.zeros((mesh_verts, M))
shats_standardized_full = np.zeros((mesh_verts, M))

y_obs_standardized_full[obs_idx,:] = y_obs_standardized
shats_null_standardized_full[obs_idx,:] = shats_null_standardized
shats_standardized_full[obs_idx,:] = shats_standardized

## Step 5: Map atlas to surface 
atlas_data = np.load("atlas/HCP_MMP1.0_Glasser.32k_fs_LR-LH.npy", allow_pickle=True).item()
labels = atlas_data["labels"]

df = pd.read_excel("atlas/Glasser_2016_Table.xlsx", index_col=None)
atlas_roi_name_dict = dict(zip(df["Parcel\nIndex"], 
							#df["Area Description"]))
							df["Area\nName"]))

## Step 6: Calculate average theta over each ROI 
theta_roi_means = {}
for seed_ROI in np.unique(labels):
	seed_roi_labels = (labels == seed_ROI)
	theta_roi_means[seed_ROI] = theta_hat[seed_roi_labels].mean().item()

theta_roi_means_sorted = sorted(theta_roi_means.items(), key=lambda kv: kv[1], reverse=True)

## Step 6: Granger Causaility 
seed_ROI = 236 ## Left Ventral Region 6
gc_seed_to_vert_BOLD, gc_vert_to_seed_BOLD = compute_gc_seed_vs_rest(y_obs_standardized_full.T, labels, seed_ROI, maxlag=2, metric="neglogp", skip_seed=True)
gc_seed_to_vert_null, gc_vert_to_seed_null = compute_gc_seed_vs_rest(shats_null_standardized_full.T, labels, seed_ROI, maxlag=2, metric="neglogp", skip_seed=True)
gc_seed_to_vert, gc_vert_to_seed = compute_gc_seed_vs_rest(shats_standardized_full.T, labels, seed_ROI, maxlag=2, metric="neglogp", skip_seed=True)


gc_seed_to_vert_BOLD[~np.isfinite(gc_seed_to_vert_BOLD)] = 0
gc_seed_to_vert_null[~np.isfinite(gc_seed_to_vert_null)] = 0
gc_seed_to_vert[~np.isfinite(gc_seed_to_vert)] = 0

gc_vert_to_seed_BOLD[~np.isfinite(gc_vert_to_seed_BOLD)] = 0
gc_vert_to_seed_null[~np.isfinite(gc_vert_to_seed_null)] = 0
gc_vert_to_seed[~np.isfinite(gc_vert_to_seed)] = 0

pval_vert_to_seed_BOLD = 10**(-gc_vert_to_seed_BOLD)
pval_vert_to_seed_null = 10**(-gc_vert_to_seed_null)
pval_vert_to_seed_corrected = 10**(-gc_vert_to_seed)

pval_seed_to_vert_BOLD = 10**(-gc_seed_to_vert_BOLD)
pval_seed_to_vert_null = 10**(-gc_seed_to_vert_null)
pval_seed_to_vert_corrected = 10**(-gc_seed_to_vert)

## FDR corrections 
alpha = 0.05

reject_bold_vert_to_seed, p_fdr_bold_vert_to_seed = fdrcorrection(pval_vert_to_seed_BOLD, alpha=alpha, method='indep')
reject_null_vert_to_seed, p_fdr_null_vert_to_seed = fdrcorrection(pval_vert_to_seed_null, alpha=alpha, method='indep')
reject_corrected_vert_to_seed, p_fdr_corrected_vert_to_seed = fdrcorrection(pval_vert_to_seed_corrected, alpha=alpha, method='indep')

psig_roi_vert_to_seed_bold = proportion_sig_by_roi(labels, reject_bold_vert_to_seed, seed_ROI=seed_ROI)
psig_roi_vert_to_seed_canonical = proportion_sig_by_roi(labels, reject_null_vert_to_seed, seed_ROI=seed_ROI)
psig_roi_vert_to_seed_corrected = proportion_sig_by_roi(labels, reject_corrected_vert_to_seed, seed_ROI=seed_ROI)


ec_thresh = 0.50
atlas_roi_name_dict[seed_ROI-180]
roi_seed_to_vert_bold_name = [e[0] for e in psig_roi_vert_to_seed_bold if e[1]>ec_thresh]
roi_seed_to_vert_bold_name
roi_seed_to_vert_canonical_name = [e[0] for e in psig_roi_vert_to_seed_canonical if e[1]>ec_thresh]
roi_seed_to_vert_canonical_name
roi_seed_to_vert_corrected_name = [e[0] for e in psig_roi_vert_to_seed_corrected if e[1]>ec_thresh]
roi_seed_to_vert_corrected_name

np.save("results/seed_%s_FC_analysis.npy"%seed_ROI,
						{"coordinates":verts, 
						"faces":tris, 
						"BOLD":[(reject_bold_vert_to_seed.astype(int),-np.log10(p_fdr_bold_vert_to_seed)),
								],
						"Fixed":[(reject_null_vert_to_seed.astype(int),-np.log10(p_fdr_null_vert_to_seed)),
								],
						"Corrected":[(reject_corrected_vert_to_seed.astype(int),-np.log10(p_fdr_corrected_vert_to_seed)),
									],
						})

##### Plot for Glassar Regions #####

plt.rcParams.update({
    "font.size": 14,          # base font size
    "axes.titlesize": 14,     # axes title
    "axes.labelsize": 14,     # x and y labels
    "xtick.labelsize": 14,    # x tick labels
    "ytick.labelsize": 14,    # y tick labels
    "legend.fontsize": 14,    # legend
    "figure.titlesize": 14,   # figure.suptitle
})

def psig_list_to_dict(psig_list, thresh):
    d = {}
    for name, prop in psig_list:
        if prop > thresh:
            clean_name = str(name).replace('\n', ' ')
            d[clean_name] = prop
    return d

ec_thresh = 0.5
d_bold       = psig_list_to_dict(psig_roi_vert_to_seed_bold,       ec_thresh)
d_canonical  = psig_list_to_dict(psig_roi_vert_to_seed_canonical,  ec_thresh)
d_corrected  = psig_list_to_dict(psig_roi_vert_to_seed_corrected,  ec_thresh)

# Union of all ROI names that pass threshold in at least one condition
all_rois = sorted(set(d_bold) | set(d_canonical) | set(d_corrected))

# Build DataFrame: 0 if ROI not present in that condition
df_roi_selected = pd.DataFrame({
    "ROI": all_rois,
    "BOLD": [d_bold.get(r, 0.0) for r in all_rois],
    "Canonical HRF": [d_canonical.get(r, 0.0) for r in all_rois],
    "Corrected HRF": [d_corrected.get(r, 0.0) for r in all_rois],
})

# Basic grouped bar plot
x = np.arange(len(df_roi_selected))      # ROI positions
width = 0.25                # bar width

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(x - width,       df_roi_selected["BOLD"],          width, label="BOLD")
ax.bar(x,               df_roi_selected["Canonical HRF"], width, label="Canonical HRF")
ax.bar(x + width,       df_roi_selected["Corrected HRF"], width, label="Corrected HRF")

ax.set_xticks(x)
ax.set_xticklabels(df_roi_selected["ROI"], rotation=45, ha="right")
ax.set_ylabel("Prop. Significant Vertices")
ax.set_ylim(0.5,1.0)
#ax.set_title(f"Effective Connectivity to Seed: {atlas_roi_name_dict[seed_ROI-180]}")
ax.set_title(f"Effective Connectivity to Seed: Ventral Area 6")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("results/strong_connected_ROI_abrev.pdf")

