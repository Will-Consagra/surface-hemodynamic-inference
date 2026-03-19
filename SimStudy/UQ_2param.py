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
import scipy
import igl

from arch.bootstrap import StationaryBootstrap, optimal_block_length

import warnings
import time 

import os 
from scipy.io import loadmat, savemat

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

HRFMODEL = "basis"

DT = 0.001
if HRFMODEL == "dbl_gamma": ## double gamma HRF model
	t1_min, t1_max = 0.5, 2.5 ## theta_1 \in [ 0.5, 2.5]
	t2_min, t2_max = 0.1, 0.3 ## theta_2 \in [0.1, 0.3]
	amin = 0.5; amax = 1.0 ## spike amplitudes 
	A_1 = torch.tensor([5.])
	A_2 = torch.tensor([15.])
	B_2 =  torch.tensor([1.])
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
			peak = self._gamma_pdf(t, A_1, theta[:,:1])
			undershoot = self._gamma_pdf(t, A_2, theta[:,:1])
			hrf = peak - theta[:,1:] * undershoot
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

elif HRFMODEL == "basis":
	t1_min, t1_max = 0.5, 1.5 ## theta_1 \in [0.5, 1.5] (% signal change)
	t2_min, t2_max = -1.0, 1.0 ## theta_2 \in [-1.0, 1.0] )(~±1s latency shift)
	amin = 1.0; amax = 1.0 ## spike amplitudes (for comparison w/ Wu 2013)
	from nipy.modalities.fmri.hrf import spmt, dspmt, gamma_params
	## Canonical HRF 2-basis model
	class HRF(nn.Module): 
		def __init__(self, t_r=1.0, T=100, L=30, onset=0.0):
			super().__init__()
			self.T = T
			self.t_r = t_r 
			self.dur = L * t_r
			self.onset = onset
			self.L = L
			t = torch.linspace(0, self.dur, int(float(self.dur) / DT)) - float(self.onset) / DT
			self.t = t[::int(self.t_r/DT)]
			self.h_t = torch.from_numpy(spmt(self.t)).float()
			self.dh_t =torch.from_numpy(dspmt(self.t)).float()
		def _compute_kernel(self, theta):
			hrf = theta[:,0:1]*self.h_t + theta[:,1:] * self.dh_t
			return hrf, self.t
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
			hrf_kernels, _ = self._compute_kernel(theta)
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
		self.t2_min = t2_min
		self.t2_max = t2_max 
		## Map range within the 99% quantile of normal
		self.eps_1 = 0.01 / 0.98 * (t1_max - t1_min) 
		self.eps_2 = 0.01 / 0.98 * (t2_max - t2_min)
		self.p_theta_tilde_1 = scipy.stats.norm(loc=0, scale=1)
		self.p_theta_tilde_2 = scipy.stats.norm(loc=0, scale=1)
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
		theta_2 = inv_link(theta_tildes[:,1], self.t2_min - self.eps_2, self.t2_max + self.eps_2)
		thetas = torch.cat((theta_1.view(-1,1), theta_2.view(-1,1)), dim=-1)
		return thetas
	def simulate(self, Nsamples):
		## simulate latents 
		theta_tildes = torch.cat((torch.from_numpy(self.p_theta_tilde_1.rvs(Nsamples)).float().view(-1,1),
									  torch.from_numpy(self.p_theta_tilde_2.rvs(Nsamples)).float().view(-1,1)),dim=-1)
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

M = 1200; TR = 0.72 ## HCP-parameters 
burn_in = 100
M_record = M; ## get to 'steady state'
M_tot = M_record + burn_in; 
start_m = M_tot-M_record
t_r = TR
L = 30
onset = 0.0

## HCP-calibration 
gamma_hat = {'sigma_e': 0.24212532107069681,
			 'lam_min': 0.011346425381130563,
			 'lam_max': 0.06959691190220721}

sigma_e = gamma_hat["sigma_e"]
lam_min, lam_max = gamma_hat["lam_min"], gamma_hat["lam_max"]

sim_model = Simulator(M_tot,
					t_r=t_r, 
					  L=L, 
					  onset=onset,
					  t1_min = t1_min, 
					  t1_max = t1_max, 
					  t2_min = t2_min, 
					  t2_max = t2_max,
					  lam_min = lam_min, 
					  lam_max = lam_max,
					   amin = amin, 
					  amax = amax, 
					  sigma_e = sigma_e)

## get surface mesh
mesh = trimesh.load("cortical_mesh/midthickness.32k_fs_LR.ply")
verts = np.array(mesh.vertices)
tris = np.array(mesh.faces)
V = verts.shape[0]

#### Set up spatial prior ####

## build inner product matrices 
G = -igl.cotmatrix(verts, tris) 
C = igl.massmatrix(verts, tris, igl.MASSMATRIX_TYPE_VORONOI) ## lumped approximation to inner product matrix
c_diag_inv = 1./C.diagonal()  # shape (n,)
C_inv = scipy.sparse.diags(c_diag_inv)

## load data 
simfname = "data/sim2data_surface.mat"
sim2data = loadmat(simfname)
y_obs = torch.from_numpy(sim2data["y_obs"]).float()
y_true = torch.from_numpy(sim2data["y_true"]).float()
neural_signals_true = torch.from_numpy(sim2data["neural_signals_true"]).float()
theta_true = torch.from_numpy(sim2data["theta_true"]).float()
theta_tilde_true = torch.from_numpy(sim2data["theta_tilde_true"]).float()


#### Learned Summary Statistic and Likelihood Emulator 

## load learned summary statistic and likelihood emulator 
fname_summary_net_state_dict = "../PreTrainedModels/models/summary_nets_2param_basis/posterior_mean_summary_statistic_2param_hrf.pth" ##pre-trained this guy
fname_lik_model_state_dict = "../PreTrainedModels/models/lik_emuls_2param_basis/likelihood_emulator_2parambasis.pth" ##pre-trained this guy

p = 2
input_channels = 2 ## real + complex channels
input_size = M_record//2 + 1
num_hidden_layers = 5
hidden_sizes = [M_record] * num_hidden_layers 
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
lik_emul = zuko.flows.NSF(p, p, transforms=5, hidden_features=[64] * 4)
lik_emul.load_state_dict(torch.load(fname_lik_model_state_dict,  map_location=torch.device("cpu"))["model_state_dict"])
lik_emul = lik_emul.to(device)
lik_emul.eval()

#### MAP INFERENCE ####
def vec2mat(datavec):
	return datavec.view(p, V).T

def mat2vec(data):
	return data.T.reshape(V*p,1)

def MAP(y, VERBOSE=False):
	def assemble_sparse_hess(H_blocks):
		# H_blocks: (V,2,2)  with entries [h11,h12; h21,h22] per v
		V = H_blocks.shape[0]
		h11 = H_blocks[:, 0, 0]
		h12 = H_blocks[:, 0, 1]
		h21 = H_blocks[:, 1, 0]
		h22 = H_blocks[:, 1, 1]
		D11 = scipy.sparse.diags(h11.cpu().detach().numpy(), 0, shape=(V, V), format="csr")
		D12 = scipy.sparse.diags(h12.cpu().detach().numpy(), 0, shape=(V, V), format="csr")
		D21 = scipy.sparse.diags(h21.cpu().detach().numpy(), 0, shape=(V, V), format="csr")
		D22 = scipy.sparse.diags(h22.cpu().detach().numpy(), 0, shape=(V, V), format="csr")
		return scipy.sparse.bmat([[D11, D12],
									[D21, D22]], format="csr")
	## define cost functions 
	def eval_cost(theta_tilde_c, S_y):
		## evaluate the augmented log-posterior density function
		## theta_tilde_c: Vp x 1 
		with torch.no_grad():
			neg_log_prob_c = -lik_emul(vec2mat(theta_tilde_c)).log_prob(S_y).sum().item()
		theta_tilde_c = theta_tilde_c.cpu().detach().numpy()
		neg_log_spat_prior_c = 0.5 * (theta_tilde_c.T @ Q_prior @ theta_tilde_c).squeeze(0).squeeze(0)
		return neg_log_prob_c, neg_log_spat_prior_c
	def ll_v(theta_tilde_v, sy_v):
		return -lik_emul(theta_tilde_v.unsqueeze(0)).log_prob(sy_v.unsqueeze(0)).sum()
	def grad_Hessian(theta_tilde_c, S_y):
		theta_tilde_c = theta_tilde_c.clone().detach().requires_grad_(True)
		neg_log_prob_c = -lik_emul(vec2mat(theta_tilde_c)).log_prob(S_y) 
		grad_neg_log_prob_c = torch.autograd.grad(
				outputs=neg_log_prob_c.sum(), 
				inputs=theta_tilde_c,
				create_graph=True
			)[0] 
		grad_c = grad_neg_log_prob_c.cpu().detach().numpy() + Q_prior @ theta_tilde_c.cpu().detach().numpy()
		Hv = torch.func.jacfwd(torch.func.jacrev(ll_v))
		H_blocks = torch.func.vmap(Hv)(vec2mat(theta_tilde_c), S_y)
		hess_neg_log_prob_c = assemble_sparse_hess(H_blocks)
		hess_c = hess_neg_log_prob_c + Q_prior
		return grad_c, hess_c
	## Step 0: pre-process data + encode observed data 
	y_stan = (y - y.mean(dim=-1).view(-1,1))/(y.std(dim=-1).view(-1,1))
	y_freq = np2freq(y_stan)
	with torch.no_grad():
		S_y = summary_network(y_freq.to(device))
	## Step 1: Initialialization w/ learned posterior mean
	theta_tilde_init = S_y.T.reshape(V*p,1)
	nlk_init, lp_init = eval_cost(theta_tilde_init, S_y)
	obj_ic = nlk_init + lp_init
	## Step 2: Newtons method 
	maxiter=100; tol=1e-5
	theta_tildes_trajectory = torch.zeros(V*p,maxiter); obj_evals = torch.zeros(maxiter,2)
	theta_tildes_trajectory[:,0:1] = theta_tilde_init
	obj_evals[0,0] = nlk_init 
	obj_evals[0,1] = lp_init
	STOP = False 
	c = 0
	#armijo line search parameters (will need to tune these)
	beta = 0.5
	sig = 1e-3 
	min_step = 1e-5
	DoArimo = True
	while not STOP:
		theta_tildes_c = theta_tildes_trajectory[:,c:(c+1)]
		## compute gradient and hessian 
		grad_c, Hessian_c = grad_Hessian(theta_tildes_c, S_y)
		## calculate newton direction 
		d_lam_c_np, info = scipy.sparse.linalg.cg(Hessian_c, grad_c)
		d_lam_c = -torch.from_numpy(d_lam_c_np).view(-1,1).float()
		#d_lam_c = -torch.from_numpy(scipy.sparse.linalg.spsolve(Hessian_c, grad_c)).view(-1,1).float()
		grad_c = torch.from_numpy(grad_c).float()
		## underrelaxation via armijo line search 
		alpha = 1
		armijo_shrink = True 
		while armijo_shrink:
			theta_tildes_c1 = theta_tildes_c + alpha*d_lam_c
			nlk_c, lp_c = eval_cost(theta_tildes_c1, S_y)
			obj_ic1 = nlk_c + lp_c
			if (obj_ic-obj_ic1> -sig*alpha*((grad_c.T @ d_lam_c).item())):
				armijo_shrink = False
			else:
				alpha = alpha*beta
				if (torch.norm(alpha*d_lam_c, p=2)<min_step): #step size too small
					armijo_shrink = False
		theta_tildes_trajectory[:,c+1] = theta_tildes_c1.squeeze(-1).detach()
		obj_evals[c+1,0] = nlk_c 
		obj_evals[c+1,1] = lp_c
		obj_ic = obj_ic1
		delta_norm_update = torch.abs(torch.sum(obj_evals[c+1,:]) - torch.sum(obj_evals[c,:]))/torch.abs(torch.sum(obj_evals[c,:]))
		if delta_norm_update < tol:
			STOP = True
		c += 1
		if c >= maxiter-1:
			STOP = True
			warnings.warn("Warning, max iterations before specified convergence tolerance.\nCurrent tolerance is%s"%delta_norm_update, UserWarning) 
		if VERBOSE:
			print(c, delta_norm_update)
	obj_evals = obj_evals[:c+1,:]
	theta_tildes_trajectory = theta_tildes_trajectory[:,:c+1]
	## estimates  
	theta_tilde_hat = theta_tildes_trajectory[:,-1].view(-1,1)
	theta_hat = mat2vec(sim_model.theta_inv_link(vec2mat(theta_tilde_hat)))
	## get grad + Hessian at the MAP 
	grad_theta_tilde_hat, Hess_theta_tilde_hat = grad_Hessian(theta_tilde_hat, S_y)
	return theta_hat, theta_tilde_hat, grad_theta_tilde_hat, Hess_theta_tilde_hat

## selected hyper-parameters from grid search in posterior_inverter_2param.py
nu = torch.tensor(1)
kappa = 0.04999999999999999
tau2 = 100000.0

## Step -1: Compute precision
## Spatial Inference 
## compute precision 
k2CG = (kappa**2) * C + G 
Q = tau2 * k2CG.T @ C_inv @ k2CG
## same smoothness for both model parameters 
Q_1 = Q
Q_2 = Q
Q_prior = scipy.sparse.block_diag((Q_1,Q_2)) 

var_rf = ((torch.lgamma(nu).exp() /(torch.lgamma(nu + 1).exp() * (4 * math.pi) * torch.tensor(kappa).pow(2 * nu))) * (1./tau2)).item()
print("Marginal Variance:",var_rf)

theta_hat_vec, theta_tilde_hat_vec, grad_theta_tilde_hat_vec, Hess_theta_tilde_hat_vec = MAP(y_obs) ## approx 2.5 - 3 mins

theta_tilde_hat = vec2mat(theta_tilde_hat_vec)
theta_hat = vec2mat(theta_hat_vec)

SqError_theta_hat = (theta_hat - theta_true)**2
MSE_theta_hat = torch.mean(SqError_theta_hat, dim=0)

##### Point Evaluation #####
print("MSE", MSE_theta_hat)
print("Normalized L2 Error", torch.norm(theta_hat - theta_true,dim=0)/torch.norm(theta_true,dim=0))
print("Bias", torch.mean(theta_hat - theta_true,dim=0))

##### UQ via Stationary Bootstrap #####
block_size = 100 ## size of blocks to resample 
R = 10 ## size of inner bootstrap blocks 
B = 100 ## size of outer bootstrap blocks (since we are distributing over 100 job_id)

## Step 1/2: Double bootstrap (this takes a long time!! should distribute on cluster)
sb_0 = StationaryBootstrap(block_size, y_obs.numpy().T)
theta_tilde_boot_estims = torch.zeros(V, B, p, 2)
for b, ((yb_t,), _) in enumerate(sb_0.bootstrap(B)):
	## Outer bootstrap
	theta_hat_b_vec, theta_tilde_hat_b_vec, grad_theta_tilde_hat_b_vec, Hess_theta_tilde_hat_b_vec = MAP(torch.from_numpy(yb_t.T).float())
	theta_tilde_hat_b = vec2mat(theta_tilde_hat_b_vec)
	## Inner bootstrap 
	sb_b = StationaryBootstrap(block_size, yb_t)
	theta_tilde_boot_b = torch.zeros(R, V, 2)
	for r, ((yb_t_b,), _) in enumerate(sb_b.bootstrap(R)):
		t1_r = time.time()
		theta_hat_b_vec_r, theta_tilde_hat_b_vec_r, grad_theta_tilde_hat_b_vec_r, Hess_theta_tilde_hat_b_vec_r = MAP(torch.from_numpy(yb_t_b.T).float())
		theta_tilde_hat_b_r = vec2mat(theta_tilde_hat_b_vec_r)
		theta_hat_b_r = vec2mat(theta_hat_b_vec_r)
		theta_tilde_boot_b[r, ...] = theta_tilde_hat_b_r
		torch.save(theta_tilde_hat_b_r, os.path.join("UQ", "2param_bootsamples", "theta_boot_%s_boot_%s.pth" % (b,r)))
	torch.save(theta_tilde_hat_b, os.path.join("UQ", "2param_bootsamples", "theta_boot_%s.pth" % (b,)))
	sd_theta_tilde_estim_b = torch.sqrt(torch.var(theta_tilde_boot_b, dim=0))
	theta_tilde_boot_estims[:,b,:,0] = theta_tilde_hat_b 
	theta_tilde_boot_estims[:,b,:,1] = sd_theta_tilde_estim_b
	print("Finished Block Bootstrap Sample",b)

sd_theta_tilde_boot = torch.sqrt(torch.var(theta_tilde_boot_estims[...,0], dim=1))

## Step 3: estimate \hat{\pi}(x,\alpha) = P{ \hat g(x) \in B^*(\alpha)}, i.e. the point-wise coverage probability, by the empirical coverage 
## of the bootstrap intervals for the initial point estimate ...

alpha0 = 0.05 # target nominal for the final band
xi = 0.05 # allow up to (xi*100)% spatial locations to be `exceptional'

alpha_grid = np.array([0.05, 0.025, 0.01, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 
						5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9, 5e-10, 1e-10, 5e-11, 1e-11, 5e-12, 1e-12,
						1e-13, 1e-14, 1e-15, 1e-25, 1e-30, 1e-35, 1e-40, 1e-45]) ## grid of alphas to explore
 
alpha_grid.sort() 

pi_counts = {a: torch.zeros_like(theta_tilde_hat, dtype=torch.float32) for a in alpha_grid}

for b in range(B):
	theta_tilde_hat_b = theta_tilde_boot_estims[:,b,:,0]; sd_b = theta_tilde_boot_estims[:,b,:,1]
	for a in alpha_grid:
		#z = scipy.stats.norm.ppf(1.0 - a/2.0)
		z = scipy.stats.norm.isf(a / 2.0)
		lower_b = theta_tilde_hat_b - z * sd_b   
		upper_b = theta_tilde_hat_b + z * sd_b
		covered_b =  ((theta_tilde_hat >= lower_b) & (theta_tilde_hat <= upper_b))
		pi_counts[a] += covered_b.float() 

pi_hat = {a: (pi_counts[a] / B) for a in alpha_grid}

## Step 4: invert point-wise intervals + build global
def invert_curve(pivals, alphas, target):
	# find first index with coverage >= target; linear interpolate
	if np.all(pivals > target):
		return float(alphas[-1])
	j = np.argmax(pivals < target)
	if j == 0:
		return float(alphas[0])
	if j >= len(alphas):
		return float(alphas[-1])
	p0, p1 = pivals[j-1], pivals[j]
	a0, a1 = alphas[j-1], alphas[j]
	w = (target - p0) / (p1 - p0 + 1e-12)
	return float((1.0 - w)*a0 + w*a1)

## ecp for each mesh vertex by alpha_grid value (V x len(alpha_grid))
Pi_1 = np.stack([pi_hat[a][:,0].cpu().numpy() for a in alpha_grid], axis=1)
Pi_2 = np.stack([pi_hat[a][:,1].cpu().numpy() for a in alpha_grid], axis=1)
#Pi = np.maximum.accumulate(np.stack([pi_hat[a].reshape(-1).cpu().numpy() for a in alpha_grid], axis=1),axis=1)

# invert coverage curve per mesh vertex to get beta_hat(v) s.t. \hat\pi(v, beta_hat) >= 1 - alpha0
target = 1.0 - alpha0
beta_hat_1 = np.full(V, fill_value=alpha_grid[-1], dtype=float)
beta_hat_2 = np.full(V, fill_value=alpha_grid[-1], dtype=float)

for v in range(V):
	beta_hat_1[v] = invert_curve(Pi_1[v, :], alpha_grid, target)
	beta_hat_2[v] = invert_curve(Pi_2[v, :], alpha_grid, target)

# get single global calibrated nominal level over space: \hat{alpha}_xi(alpha0) is the xi-quantile of beta_hat(v)
alpha_xi_1 = float(np.quantile(beta_hat_1, q=xi))  # single global calibrated nominal level
alpha_xi_2 = float(np.quantile(beta_hat_2, q=xi))  # single global calibrated nominal level

# calculate final intervals 
z_star_1 = scipy.stats.norm.isf(alpha_xi_1 / 2.0)
z_star_2 = scipy.stats.norm.isf(alpha_xi_2 / 2.0)

lb_calibrated_1 = theta_tilde_hat[:,0] - z_star_1 * sd_theta_tilde_boot[:,0]
ub_calibrated_1 = theta_tilde_hat[:,0] + z_star_1 * sd_theta_tilde_boot[:,0]

lb_calibrated_2 = theta_tilde_hat[:,1] - z_star_2 * sd_theta_tilde_boot[:,1]
ub_calibrated_2 = theta_tilde_hat[:,1] + z_star_2 * sd_theta_tilde_boot[:,1]

# map to theta space 
lb_calibrated_theta = sim_model.theta_inv_link(torch.cat((lb_calibrated_1.view(-1,1), lb_calibrated_2.view(-1,1)), dim=-1))
ub_calibrated_theta = sim_model.theta_inv_link(torch.cat((ub_calibrated_1.view(-1,1), ub_calibrated_2.view(-1,1)), dim=-1))

# evaluate coverage 
ecp_1 = ((lb_calibrated_theta[:,0] <= theta_true[:,0]) & (ub_calibrated_theta[:,0] >= theta_true[:,0])).sum(dim=0)/V
ecp_2 = ((lb_calibrated_theta[:,1] <= theta_true[:,1]) & (ub_calibrated_theta[:,1] >= theta_true[:,1])).sum(dim=0)/V

# interval length 
il_1 = (ub_calibrated_theta[:,0] - lb_calibrated_theta[:,0]).mean(dim=0)
il_2 = (ub_calibrated_theta[:,1] - lb_calibrated_theta[:,1]).mean(dim=0)

# (Optional) also keep the per-voxel beta_hat and the calibrated nominal level
print(f"H&H calibrated empirical coverage theta 1: {ecp_1}  (target alpha0 = {alpha0})")
print(f"H&H Interval Length theta 1: {il_1}")

# (Optional) also keep the per-voxel beta_hat and the calibrated nominal level
print(f"H&H calibrated empirical coverage theta 1: {ecp_2}  (target alpha0 = {alpha0})")
print(f"H&H Interval Length theta 1: {il_2}")


