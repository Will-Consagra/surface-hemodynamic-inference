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
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

## sparse cholesky to avoid memory blow up
def sample_from_sparse_precision(Q_sp, z, jitter=0.0):
    Q_csc = scipy.sparse.csc_matrix(Q_sp)
    factor = cholesky(Q_csc, beta=jitter)
    L = factor.L().tocsc()
    x_perm = scipy.sparse.linalg.spsolve_triangular(L.T, z, lower=False)
    x = factor.apply_Pt(x_perm)
    return np.asarray(x).reshape(-1, 1)

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

## p(theta) ~ exp(low-freq-GP)

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

## define smooth HRF prior 
kappa_1_true = 5e-2; tau2_1_true = 1e4
kappa_2_true = 5e-2; tau2_2_true = 1e4

nu_true = torch.tensor(1)
var_rf_1_true = ((torch.lgamma(nu_true).exp() /(torch.lgamma(nu_true + 1).exp() * (4 * math.pi) * torch.tensor(kappa_1_true).pow(2 * nu_true))) * (1./tau2_1_true)).item()
var_rf_2_true = ((torch.lgamma(nu_true).exp() /(torch.lgamma(nu_true + 1).exp() * (4 * math.pi) * torch.tensor(kappa_2_true).pow(2 * nu_true))) * (1./tau2_2_true)).item()
print(var_rf_1_true)
print(var_rf_2_true)

Q_1_true_sp = tau2_1_true * ((kappa_1_true**2) * C + G).T @ C_inv @ ((kappa_1_true**2) * C + G)
Q_2_true_sp = tau2_2_true * ((kappa_2_true**2) * C + G).T @ C_inv @ ((kappa_2_true**2) * C + G)

simfname = "data/sim2data_surface.mat"
if not os.path.exists(simfname):
	np.random.seed(0)
	z1 = np.random.randn(V,1); z2 = np.random.randn(V,1)
	theta_tilde_true_1 = torch.from_numpy(sample_from_sparse_precision(Q_1_true_sp,z1)).float().to(device) 
	theta_tilde_true_2 = torch.from_numpy(sample_from_sparse_precision(Q_2_true_sp,z2)).float().to(device) 
	theta_tilde_true = torch.cat((0.5*theta_tilde_true_1, theta_tilde_true_2), dim=-1)
	theta_true = sim_model.theta_inv_link(theta_tilde_true)
	var_rf_1 = (torch.lgamma(nu_true).exp() /(torch.lgamma(nu_true + 1).exp() * (4 * math.pi) * torch.tensor(kappa_1_true).pow(2 * nu_true))) * (1./tau2_1_true)
	var_rf_2 = (torch.lgamma(nu_true).exp() /(torch.lgamma(nu_true + 1).exp() * (4 * math.pi) * torch.tensor(kappa_2_true).pow(2 * nu_true))) * (1./tau2_2_true)
	## load marginal kernel inverter (trained in inverter.ipynb)  
	y_true = torch.zeros((V, M_tot))
	neural_signals_true = torch.zeros((V, M_tot))
	chunk_size = 1000
	for v0 in range(0, V, chunk_size):
		v1 = min(v0 + chunk_size, V)
		## p(s) ~ ((eventually simulate some basic correlation structure, for now iid))
		_, _, _, neural_signals_true_v = sim_model.simulate(v1 - v0)
		## p(y|s,theta) 
		H_theta_true_v = sim_model.hrf_model(theta_true[v0:v1,:])
		y_true_v = torch.matmul(H_theta_true_v, neural_signals_true_v.unsqueeze(-1)).squeeze(-1)
		## store 
		neural_signals_true[v0:v1,:] = neural_signals_true_v
		y_true[v0:v1,:] = y_true_v
		print(v0,v1)
	y_obs = y_true + sigma_e*torch.normal(0, 1, size=(V, M_tot))
	## cut first 100s to get steady state signals 
	y_obs = y_obs[:,start_m:]
	y_true = y_true[:,start_m:]
	neural_signals_true = neural_signals_true[:,start_m:]
	#### Save data ####
	hrf_kernels_theta_true, _ = sim_model.hrf_model._compute_kernel(theta_true)
	sim2data = {
		"y_obs":y_obs.cpu().detach().numpy(),
		"y_true":y_true.cpu().detach().numpy(),
		"neural_signals_true":neural_signals_true.cpu().detach().numpy(),
		"theta_true":theta_true.cpu().detach().numpy(),
		"theta_tilde_true":theta_tilde_true.cpu().detach().numpy(),
		"hrf_kernels_true":hrf_kernels_theta_true.cpu().detach().numpy(),
		"TR":t_r,
		"sigma_e":sigma_e,
	}
	savemat(simfname, sim2data)
else:
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

## pre-process data 
## Normalize signals
y_obs_standardized = (y_obs - y_obs.mean(dim=-1).view(-1,1))/(y_obs.std(dim=-1).view(-1,1))
## FT
y_obs_standardized_freq = np2freq(y_obs_standardized)

#### MAP INFERENCE ####

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

def vec2mat(datavec):
	return datavec.view(p, V).T

def mat2vec(data):
	return data.T.reshape(V*p,1)

def sparse_slogdet_from_superlu(splu):
    """
    Computes the sign and the logarithm of the determinant of a sparse matrix from its
    SuperLU decomposition.
    References
    ----------
    This function is based on the following GIST and its discussion:
    https://gist.github.com/luizfelippesr/5965a536d202b913beda9878a2f8ef3e

    """
    ### Auxiliary Function ###
    def count_min_num_swaps_to_sort(arr: np.ndarray):
        """
        Minimum number of swaps needed to order a permutation array.

        """
        # from https://www.thepoorcoder.com/hackerrank-minimum-swaps-2-solution/
        a = dict(enumerate(arr))
        b = {v: k for k, v in a.items()}
        count = 0
        for i in a:
            x = a[i]
            if x != i:
                y = b[i]
                a[y] = x
                b[x] = y
                count += 1
        return count
    ### Main Part ###
    diagU = splu.U.diagonal()
    logabsdet = np.log(np.abs(diagU)).sum()
    fact_sign = -1 if np.count_nonzero(diagU < 0.0) % 2 == 1 else 1
    row_sign = -1 if count_min_num_swaps_to_sort(splu.perm_r) % 2 == 1 else 1
    col_sign = -1 if count_min_num_swaps_to_sort(splu.perm_c) % 2 == 1 else 1
    # col_sign = 1 # <-- If this is uncommented, this produces the `perm_r`-only code
    sign = -1.0 if fact_sign * row_sign * col_sign < 0 else 1.0
    return sign, logabsdet

def laplace_log_evidence(theta_tilde_map, LU=False):
	ll_map = lik_emul(vec2mat(theta_tilde_map)).log_prob(S_y).sum()  # torch scalar
	ll_map_val = float(ll_map.detach().cpu().item())
	th_np = theta_tilde_map.detach().cpu().numpy()  # shape (V,1)
	quad = 0.5 * float((th_np.T @ (Q_prior @ th_np)).squeeze())
	logdet_Q = float(cholesky(Q_prior).logdet())
	_, H_sp = grad_Hessian(theta_tilde_map, S_y)  # (grad, H) ; H is scipy.sparse
	if LU:
		lu = scipy.sparse.linalg.splu(H_sp)
		sign_logdet_H, abs_logdet_H = sparse_slogdet_from_superlu(lu)
		logdet_H = sign_logdet_H * abs_logdet_H
	else:
		logdet_H = float(cholesky(H_sp).logdet())
	# ll_map - 0.5 θ^T Q θ + 0.5 log|Q| - 0.5 log|H|
	log_evidence = ll_map_val - quad + 0.5 * (logdet_Q - logdet_H)
	return log_evidence, {
		"ll_map": ll_map_val,
		"prior_quad": quad,
		"logdet_Q": logdet_Q,
		"logdet_H": logdet_H,
	}


## define smooth HRF prior 
nu = torch.tensor(1)

## hyper-parameter grid 
marg_kappa_grid = np.geomspace(5e-5, 5e-1, num=5)   
marg_tau2_grid  = np.geomspace(1e2,  1e6,  num=5) 
hyperparm_grid = [(mk, mt) for mk in marg_kappa_grid for mt in marg_tau2_grid]

inference = {}

for hi, (kappa, tau2) in enumerate(hyperparm_grid):
	var_rf = ((torch.lgamma(nu).exp() /(torch.lgamma(nu + 1).exp() * (4 * math.pi) * torch.tensor(kappa).pow(2 * nu))) * (1./tau2)).item()
	print("Marginal Variance:",var_rf)
	## Step -1: Compute precision
	## Spatial Inference 
	## compute precision 
	k2CG = (kappa**2) * C + G 
	Q = tau2 * k2CG.T @ C_inv @ k2CG
	## same smoothness for both model parameters 
	Q_1 = Q
	Q_2 = Q
	Q_prior = scipy.sparse.block_diag((Q_1,Q_2)) 
	## Step 0: Encode observed data 
	S_y = summary_network(y_obs_standardized_freq.to(device))
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
		d_lam_c = -torch.from_numpy(scipy.sparse.linalg.spsolve(Hessian_c, grad_c)).view(-1,1).float()
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
		theta_tildes_trajectory[:,c+1] = theta_tildes_c1.squeeze(-1)
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
	obj_evals = obj_evals[:c+1,:]
	theta_tildes_trajectory = theta_tildes_trajectory[:,:c+1]
	## compute (approx) marginal likelihood 
	theta_tilde_map = theta_tildes_trajectory[:,-1].view(-1,1)
	try:
		log_evidence, parts = laplace_log_evidence(theta_tilde_map)
	except Exception as e:
		log_evidence = -np.inf 
	inference[hi] = {"kappa":kappa, 
					"tau2":tau2, 
					"lml": log_evidence, 
					"theta_tildes_trajectory":theta_tildes_trajectory
					}
	print(hi, "Finished Kappa,", kappa, "Tau2,", tau2, "Marginal Lik:", log_evidence)

hi_max, _ = sorted([(hi, inference[hi]["lml"]) for hi in inference.keys()], key=lambda x:x[1])[-1]

kappa = inference[hi_max]["kappa"]
tau2 = inference[hi_max]["tau2"]
lml = inference[hi_max]["lml"]
theta_tildes_trajectory = inference[hi_max]["theta_tildes_trajectory"]


theta_tilde_hat = vec2mat(theta_tildes_trajectory[:,-1])
theta_hat = sim_model.theta_inv_link(theta_tilde_hat)

hrf_kernels_theta_estim, _ = sim_model.hrf_model._compute_kernel(theta_hat)


##### Evaluation #####

## in the parameter space 
print("g(theta)")
print("Spatial Inference Results")
print("Normalized L2 Error", torch.mean((theta_tilde_hat - theta_tilde_true)**2,dim=0)/torch.mean(theta_tilde_true**2, dim=0))
print("Bias", torch.mean(theta_tilde_hat - theta_tilde_true, dim=0))

print("theta")
print("Spatial Inference Results")
print("L2 Error", torch.mean((theta_hat - theta_true)**2,dim=0))
print("Normalized L2 Error", torch.mean((theta_hat - theta_true)**2,dim=0)/torch.mean(theta_true**2, dim=0))
print("Bias", torch.mean(theta_hat - theta_true, dim=0))

evalfname = "data/sim2data_inference_proposed.mat"
savemat(evalfname, {"hrf_kernel_estim":hrf_kernels_theta_estim.cpu().detach().numpy(),
					"theta_hat":theta_hat.cpu().detach().numpy(),
					"theta_tilde_hat":theta_tilde_hat.cpu().detach().numpy()})
