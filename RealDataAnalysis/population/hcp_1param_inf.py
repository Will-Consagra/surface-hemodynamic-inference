##conda activate fmri_inv_rda
import numpy as np 
import math
import torch 
import torch.nn as nn
from torch.special import gammaln, xlogy
from torch.utils.data import DataLoader, Dataset
import zuko 

import scipy
from scipy.fft import fft 
from torch.distributions.multivariate_normal import MultivariateNormal

import trimesh
import igl
import scipy
from sksparse.cholmod import cholesky

import warnings

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
C_12 = torch.tensor([P_U_RATIO])

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
		hrf = peak - C_12 * undershoot
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

def link(theta, min_theta, max_theta):
	u = (theta - min_theta)/(max_theta - min_theta)
	return torch.distributions.Normal(0,1).icdf(u)

def inv_link(theta_tilde, min_theta, max_theta):
	u = torch.distributions.Normal(0, 1).cdf(theta_tilde)
	return min_theta + (max_theta - min_theta) * u

def np2freq(sigs):
	sigs_ft = torch.fft.rfft(sigs, dim=-1)
	sigs_real = sigs_ft.real
	sigs_complex = sigs_ft.imag
	sigs_freq = torch.stack([sigs_real, sigs_complex], dim=1)
	return sigs_freq

class Simulator:
	def __init__(self, M, t_r=1.0, L=30, onset=0.0, t1_min = 5, t1_max = 7, 
				 t2_min = 0.5, t2_max = 1.5, lam_min = 0.05, lam_max = 0.1,
				   amin = 0.5, amax = 1.0, sigma_e = 2.5e-2):
		super().__init__()
		#### Measurement Parameters ####
		self.M = M
		self.t_r=t_r ##T_r of the acquisition 
		self.hrf_model = HRF(t_r=t_r, T=M, L=L, onset=onset)   
		#### Hierarchical Model ####
		self.t1_min = t1_min
		self.t1_max = t1_max
		## Map range within the 99% quantile of normal
		self.eps_1 = 0.01 / 0.98 * (t1_max - t1_min) 
		self.p_theta_tilde_1 = scipy.stats.norm(loc=0, scale=1)
		self.p_lambda = scipy.stats.uniform(loc=lam_min, scale=lam_max-lam_min)
		self.p_amp = scipy.stats.uniform(loc=amin, scale=amax-amin)
		self.sigma_e = sigma_e
	def simulate_neural_signals(self, lambdas):
		Nsamples = len(lambdas)
		signal_map = {}
		for ns in range(Nsamples):
			lambda_rate_ns = lambdas[ns]
			t = 0
			event_times = []; amplitudes = []
			while t < (self.M*self.t_r):
				# Draw the next inter-arrival time from Exponential(lambda)
				dt = np.random.exponential(scale=1/lambda_rate_ns)
				at = self.p_amp.rvs(1).item()
				t += dt
				if t < (self.M*self.t_r):
					event_times.append(t)
					amplitudes.append(at)
			signal_map[ns] = (np.array(event_times), np.array(amplitudes))
		return signal_map
	def theta_inv_link(self, theta_tildes):
		theta_1 = inv_link(theta_tildes[:,0], self.t1_min - self.eps_1, self.t1_max + self.eps_1)
		return theta_1.view(-1,1)
	def simulate(self, Nsamples):
		## simulate latents 
		theta_tildes = torch.from_numpy(self.p_theta_tilde_1.rvs(Nsamples)).float().view(-1,1)
		thetas = self.theta_inv_link(theta_tildes)
		lambdas = self.p_lambda.rvs(Nsamples)
		signal_map = self.simulate_neural_signals(lambdas) 
		discretized_neural_signals = np.zeros((Nsamples, self.M))
		for ns in range(Nsamples):
			spike_times_ns = signal_map[ns][0]; spike_amps_ns = signal_map[ns][1];
			if spike_times_ns.size > 0:
				for t, a_t in zip(spike_times_ns, spike_amps_ns):
					t_idx = int(t // self.t_r)
					discretized_neural_signals[ns, t_idx] = discretized_neural_signals[ns, t_idx] + a_t 
		discretized_neural_signals = torch.from_numpy(discretized_neural_signals).float()
		## simulate observed signals 
		H_theta = self.hrf_model(thetas)
		y_true = torch.matmul(H_theta, discretized_neural_signals.unsqueeze(-1)).squeeze(-1)
		y_obs = y_true + self.sigma_e*torch.normal(0, 1, size=(Nsamples, self.M))
		return y_obs, y_true, theta_tildes, discretized_neural_signals

## learn summary network (approximate marginal posterior mean S(y) \approx E[theta|y]
class MLP(nn.Module):
	def __init__(self, input_channels, input_size, hidden_sizes, output_size, activation=nn.ReLU(), dropout=0.0):
		super().__init__()
		layers = []
		# Add the first linear layer to handle input channels
		layers.append(nn.Linear(input_channels * input_size, hidden_sizes[0]))
		layers.append(activation)
		if dropout > 0.0:
			layers.append(nn.Dropout(dropout))
		for i in range(1, len(hidden_sizes)):
			layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
			layers.append(activation)
			if dropout > 0.0:
				layers.append(nn.Dropout(dropout))
		layers.append(nn.Linear(hidden_sizes[-1], output_size))
		self.mlp = nn.Sequential(*layers)
	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.mlp(x) 

## Marginal MAP Estimator ## 
def MAP(y_freq, kappa, tau2):
	## Spatial Inference 
	def eval_cost(theta_tilde_c):
		## evaluate the augmented log-posterior density function here ....
		neg_log_prob_c = -lik_emul(theta_tilde_c).log_prob(S_y).sum()
		theta_tilde_c = theta_tilde_c.cpu().detach().numpy()
		neg_log_spat_prior_c = 0.5 * (theta_tilde_c.T @ Q_prior_sp @ theta_tilde_c).squeeze(0).squeeze(0)
		return neg_log_prob_c, neg_log_spat_prior_c
	def grad_Hessian(theta_tilde_c):
		theta_tilde_c = theta_tilde_c.clone().detach().requires_grad_(True)
		neg_log_prob_c = -lik_emul(theta_tilde_c).log_prob(S_y) 
		grad_neg_log_prob_c = torch.autograd.grad(
				outputs=neg_log_prob_c.sum(), 
				inputs=theta_tilde_c,
				create_graph=True,
			)[0]  
		hess_neg_log_prob_c = torch.autograd.grad(
			outputs=grad_neg_log_prob_c.sum(),  # sum over N to get scalar again
			inputs=theta_tilde_c,
			retain_graph=True
		)[0]
		grad_c = grad_neg_log_prob_c.cpu().detach().numpy() + Q_prior_sp @ theta_tilde_c.cpu().detach().numpy()
		## create sparse diagonal tensor (numpy)
		hess_c = scipy.sparse.diags(hess_neg_log_prob_c.squeeze(-1).cpu().detach().numpy()) + Q_prior_sp
		return grad_c, hess_c
	## Step -1: Compute precision
	k2CG = (kappa**2) * C + G 
	Q = tau2 * k2CG.T @ Cinv @ k2CG
	Q_prior_sp = Smat @ Q @ Smat.T 
	## Step 0: Encode data 
	## pre-process data 
	S_y = summary_network(y_freq.to(device))
	## Step 1: Initialialization w/ learned posterior mean
	theta_tilde_init = S_y
	nlk_init, lp_init = eval_cost(theta_tilde_init)
	## Step 2: Newtons method 
	maxiter=100; tol=1e-6
	theta_tildes_trajectory = torch.zeros(V,maxiter); obj_evals = torch.zeros(maxiter,2)
	theta_tildes_trajectory[:,0:1] = theta_tilde_init
	obj_evals[0,0] = nlk_init 
	obj_evals[0,1] = lp_init
	STOP = False 
	c = 0
	#armijo line search parameters (tune these)
	beta = 0.5
	sig = 1e-3 
	min_step = 1e-5
	DoArimo = True
	while not STOP:
		theta_tildes_c = theta_tildes_trajectory[:,c:(c+1)]
		## compute gradient and hessian 
		grad_c, Hessian_c = grad_Hessian(theta_tildes_c)
		## calculate newton direction 
		#d_lam_c = -torch.from_numpy(np.linalg.solve(Hessian_c.cpu().detach().numpy(), grad_c.cpu().detach().numpy()))
		d_lam_c = -torch.from_numpy(scipy.sparse.linalg.spsolve(Hessian_c, grad_c)).view(-1,1).float()
		grad_c = torch.from_numpy(grad_c).float()
		#d_lam_c_sp, exit_code = -sp.linalg.cg(sp.csr_matrix(Hessian_c.cpu().detach().numpy()), grad_c.cpu().detach().numpy()) 
		alpha = 1
		theta_tildes_c1 = theta_tildes_c + alpha*d_lam_c
		nlk_c, lp_c = eval_cost(theta_tildes_c1)
		if DoArimo: ## underrelaxation via armijo line search
			armijo_shrink = True 
			obj_ic = torch.sum(obj_evals[c,:])
			while armijo_shrink:
				theta_tildes_c1 = theta_tildes_c + alpha*d_lam_c
				nlk_c, lp_c = eval_cost(theta_tildes_c1)
				obj_ic1 = nlk_c + lp_c
				if (obj_ic-obj_ic1> -sig*alpha*((grad_c.T @ d_lam_c).item())):
					armijo_shrink = False
				else:
					alpha = alpha*beta
					if (torch.norm(alpha*d_lam_c, p=2)<min_step): #step size too small
						armijo_shrink = False
						#STOP = True
						#print("Step size too small", k) 
		theta_tildes_trajectory[:,c+1] = theta_tildes_c1.squeeze(-1)
		obj_evals[c+1,0] = nlk_c 
		obj_evals[c+1,1] = lp_c
		delta_norm_update = torch.abs(torch.sum(obj_evals[c+1,:]) - torch.sum(obj_evals[c,:]))/torch.abs(torch.sum(obj_evals[c,:]))
		if delta_norm_update < tol:
			STOP = True
		c += 1
		if c >= maxiter-1:
			STOP = True
			warnings.warn("Warning, max iterations before specified convergence tolerance.\nCurrent tolerance is%s"%delta_norm_update, UserWarning) 
	obj_evals = obj_evals[:c+1,:]
	theta_tildes_trajectory = theta_tildes_trajectory[:,:c+1]
	## estimates  
	theta_tilde_hat = theta_tildes_trajectory[:,-1].view(-1,1)
	theta_hat = sim_model.theta_inv_link(theta_tilde_hat)
	## get grad + Hessian at the MAP 
	grad_theta_tilde_hat, Hess_theta_tilde_hat = grad_Hessian(theta_tilde_hat)
	return theta_hat, theta_tilde_hat, grad_theta_tilde_hat, Hess_theta_tilde_hat

##### UQ via Laplace #####
def sample_approx_posterior_Cholesky(theta_tilde_MAP, Hessian_theta_tilde_MAP, Nsamples=100): 
	L_MAP = cholesky(Hessian_theta_tilde_MAP).L()
	theta_tilde_samples = torch.zeros(Nsamples, V, p)
	theta_samples = torch.zeros(Nsamples, V, p)
	for n in range(Nsamples):
		z = np.random.randn(V);
		theta_tilde_sample_centered = torch.from_numpy(scipy.sparse.linalg.spsolve(L_MAP.T, z)).float().view(-1,1).to(device) 
		theta_tilde_sample = theta_tilde_MAP + theta_tilde_sample_centered
		theta_sample = sim_model.theta_inv_link(theta_tilde_sample)
		theta_tilde_samples[n,...] = theta_tilde_sample
		theta_samples[n,...] = theta_sample
		print(n)
	return theta_tilde_samples, theta_samples


#### Instantiate Simulator ####
M = 1200; TR = 0.72
burn_in = 100
M_record = M; ## get to 'steady state'
M_tot = M_record + burn_in; 
start_m = M_tot-M_record
t_r = TR
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

sim_model = Simulator(M_tot,
					t_r=t_r, 
					  L=L, 
					  onset=onset,
					  t1_min = t1_min, 
					  t1_max = t1_max, 
					  lam_min = lam_min, 
					  lam_max = lam_max,
					   amin = amin, 
					  amax = amax, 
					  sigma_e = sigma_e)

## load learned summary statistic and likelihood emulator 
fname_summary_net_state_dict = "../../PreTrainedModels/models/summary_nets_1param/posterior_mean_summary_statistic_shifted_hrf.pth" ##pre-trained this guy
fname_lik_model_state_dict = "../../PreTrainedModels/models/lik_emul_1param/likelihood_emulator_posterior_mean_summary_shifted_hrf.pth" ##pre-trained this guy

## Summay Statistic 
p = 1
input_channels = 2 ## real + complex channels
input_size = M//2 + 1
hidden_sizes = [M, M//2, M//4]
activation = nn.ReLU()
dropout = 0.0
summary_network = MLP(input_channels, 
					  input_size, 
					  hidden_sizes, 
					  p, 
					  activation=activation, dropout=dropout)
summary_network.load_state_dict(torch.load(fname_summary_net_state_dict, map_location=torch.device("cpu"))["model_state_dict"])
summary_network = summary_network.to(device)
summary_network.eval()

## Likelihood Emulator 
lik_emul = zuko.flows.NSF(p, p, transforms=5, hidden_features=[64] * 3)
lik_emul.load_state_dict(torch.load(fname_lik_model_state_dict,  map_location=torch.device("cpu"))["model_state_dict"])
lik_emul.eval()

## pre-selected HRF prior on 103818
nu = torch.tensor(1)
kappa = 1e-3
tau2 = 1e3

#### Process fMRI Signals ####
subj_ids = np.unique([fname.split(".")[0] for fname in os.listdir("S1200_Rest3TRecommended")])

for subj_id in subj_ids:
	SIGNAL_FILE = os.path.join("S1200_Rest3TRecommended", "%s.L.1D"%(subj_id,))
	y_obs = torch.from_numpy(np.loadtxt(SIGNAL_FILE)).float()
	#t = torch.tensor([i*TR for i in range(y_obs.shape[1])])

	## z-score 
	mesh_verts, M = y_obs.shape

	## get mask (caused by mesh hole)
	filter_vertices = y_obs.std(dim=-1) == 0
	cc_mask = (~filter_vertices).to(torch.bool)         
	obs_idx = torch.nonzero(cc_mask, as_tuple=True)[0]  
	V = len(obs_idx)

	## linear selector 
	Smat = scipy.sparse.csr_matrix((V, mesh_verts), dtype=np.float32) 
	for i, vix in enumerate(obs_idx):
		Smat[i, vix] = 1.

	#### Set up spatial prior ####
	## get surface mesh
	MESH_FILE = "S1200_StructuralRecommended/%s.L.midthickness_MSMAll.32k_fs_LR.ply" % (subj_id,)
	mesh = trimesh.load(MESH_FILE)
	verts = np.array(mesh.vertices)
	tris = np.array(mesh.faces)

	## LBO eigenfunctions (if want low-rank instead of sparse)
	#rhos, phi = scipy.sparse.linalg.eigsh(A=G, M=C, k=50, sigma=0.0, which="LM")

	## build inner product matrices 
	G = -igl.cotmatrix(verts, tris) 
	C = igl.massmatrix(verts, tris, igl.MASSMATRIX_TYPE_VORONOI) ## lumped approximation to inner product matrix
	c_diag_inv = 1./C.diagonal()  # shape (n,)
	Cinv = scipy.sparse.diags(c_diag_inv)

	## signals -> frequency domian 
	y_obs_standard = (y_obs - y_obs.mean(dim=-1).view(-1,1))/(y_obs.std(dim=-1).view(-1,1))
	y_obs_freq = np2freq(y_obs_standard[obs_idx,...])

	## MAP estimate 
	theta_hat, theta_tilde_hat, grad_theta_tilde_hat, Hess_theta_tilde_hat = MAP(y_obs_freq, kappa, tau2)
	theta_hat_full = torch.zeros(mesh_verts, p); theta_tilde_hat_full = torch.zeros(mesh_verts, p);
	theta_hat_full[obs_idx,:] = theta_hat; theta_tilde_hat_full[obs_idx,:] = theta_tilde_hat

	OUT_FILE = "S1200_RestT3_Estimates/%s_1param_estims.npy" % (subj_id,)

	np.save(OUT_FILE, {"coordinates":verts, 
						"faces":tris, 
						"theta_hat":theta_hat_full.cpu().detach().numpy(),
						"theta_tilde_hat":theta_tilde_hat_full.cpu().detach().numpy(),
						"hyperparms":[kappa, tau2]})
	print("Finished", OUT_FILE)

