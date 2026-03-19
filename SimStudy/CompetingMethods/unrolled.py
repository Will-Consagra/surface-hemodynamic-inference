##conda activate neuro_proc
import numpy as np
import torch
import torch.nn as nn
from torch.special import gammaln, xlogy
from torch.utils.data import DataLoader, Dataset

import scipy
from scipy.fft import fft 
from scipy.special import sph_harm, legendre, gamma
from torch.distributions.multivariate_normal import MultivariateNormal

import os 
from scipy.io import loadmat
import time 

import matplotlib.pyplot as plt


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### Unrolled Network ####

class ProximalBlock(nn.Module):
	def __init__(self, input_dim, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
		super().__init__()
		conv_output_dim = (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
		self.conv1d = nn.Conv1d(1, out_channels, kernel_size, stride=stride)
		self.relu = nn.ReLU(inplace=True)
		self.fc1 = nn.Linear(out_channels*conv_output_dim, input_dim)
	def forward(self,x):
		x = self.conv1d(x)
		x = self.relu(x)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		return x

class LPGD(nn.Module):
	"""
	Vanilla Unrolled Projected Gradient Descent Algorithm
	"""
	def __init__(self, n, max_iter, gamma_init=5e-2, hidden_features=100, kernel_size=3, stride=1, padding=0, dilation=1):
		"""
		EH: array, E_{p(theta)}[H(theta)]

		"""
		super().__init__()
		self.net = nn.ModuleList([ProximalBlock(n, hidden_features, 
								kernel_size=kernel_size, stride=stride, 
								padding=padding, dilation=dilation) for k in range(max_iter)])
		self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float64), requires_grad=True) ## this can also be per layer ...
	def forward(self, y, H):
		## Step 1: initialize x_0 = H'y 
		x = torch.matmul(H.transpose(1, 2), y.unsqueeze(2)).squeeze(-1)
		## Step 2:
		for k, layer in enumerate(self.net):
			term1 = (torch.matmul(H, x.unsqueeze(2)).squeeze(-1) - y).unsqueeze(2)
			term2 = torch.matmul(H.transpose(1, 2), term1).squeeze(-1)
			term3 = self.gamma*term2
			x = layer((x - term3).unsqueeze(1))
		return x

class UnrolledReLU(nn.Module):
	def __init__(self, n, max_iter, gamma_init=5e-2):
		"""
		EH: array, E_{p(theta)}[H(theta)]

		"""
		super().__init__()
		layers = []
		for _ in range(max_iter - 1):
			layers += [nn.Linear(n, n), nn.ReLU(inplace=True)]
		layers += [nn.Linear(n, n)]
		self.net = nn.Sequential(*layers)
		self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float64), requires_grad=True) 
		self.max_iter = max_iter 
	def forward(self, y, H):
		## Step 1: initialize x_0 = H'y 
		x = torch.matmul(H.transpose(1, 2), y.unsqueeze(2)).squeeze(-1)
		## Step 2:
		#for k, layer in enumerate(self.net):
		for k in range(self.max_iter):
			term1 = (torch.matmul(H, x.unsqueeze(2)).squeeze(-1) - y).unsqueeze(2)
			grad = torch.matmul(H.transpose(1, 2), term1).squeeze(-1)
			term3 = self.gamma*grad
			#x = layer((x - term3))
			x = self.net(x - term3)
		return x

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

def link(theta, min_theta, max_theta):
	u = (theta - min_theta)/(max_theta - min_theta)
	return torch.distributions.Normal(0,1).icdf(u)

def inv_link(theta_tilde, min_theta, max_theta):
	u = torch.distributions.Normal(0, 1).cdf(theta_tilde)
	return min_theta + (max_theta - min_theta) * u

class Simulator:
	def __init__(self, M, t_r=1.0, L=30, onset=0.0, t1_min = 5, t1_max = 7, 
				 t2_min = 0.5, t2_max = 1.5, lam_min = 0.05, lam_max = 0.1,
				   amin = 0.5, amax = 1.0, sigma_e = 2.5e-2):
		super().__init__()
		#### Measurement Parameters ####
		self.M = M
		self.t_r=t_r ##T_r of the acquisition 
		self.L=L ##kernel length  
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

MODELDIR = "../../PreTrainedModels/models/unrolled_1param"

M = 1200; TR = 0.72 ## HCP-parameters 
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
 'lam_max': 0.2167510244193856}

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


## evaluation 
np.random.seed(0)
N_test = 200
Npost_samples = 5000

y_eps_test, y_true_test, theta_tildes_test, neural_signals_test = sim_model.simulate(N_test)
y_eps_test = y_eps_test[:,start_m:]
y_true_test = y_true_test[:,start_m:]
neural_signals_test = neural_signals_test[:,start_m:]
theta_test = sim_model.theta_inv_link(theta_tildes_test)
H_theta_test = sim_model.hrf_model(theta_test)
H_theta_test = H_theta_test[...,start_m:,start_m:]

##### Unrolled Optimizer Architecture ##### 
num_layers = 5; hidden_features=100; 
gamma_init = torch.tensor([0.02])

#kernel_size=3; stride=1; padding=0; dilation=1
#gamma_init = (1 / torch.norm(H, p="fro", dim=(1,2)) ** 2).mean()
#net = LPGD(M, num_layers, gamma_init=gamma_init, hidden_features=hidden_features, 
#				kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
net = UnrolledReLU(M, num_layers, gamma_init=gamma_init)
net = net.float().to(device)

##### Unrolled Optimizer Training ##### 
Nbatch = 200
lr = 1e-5 
optimizer = torch.optim.Adam(net.parameters(), 
							lr=lr)

num_epochs_track = 1000
num_epochs = 500000

for epoch in range(num_epochs):
	t1 = time.time()
	## noisy BOLD  
	batch_y_eps, batch_y_true, batch_theta_tildes, batch_neural_signals = sim_model.simulate(Nbatch)
	batch_y_true = batch_y_true[:,start_m:]
	batch_neural_signals = batch_neural_signals[:,start_m:]
	batch_y_eps = batch_y_eps[:,start_m:]
	batch_thetas = sim_model.theta_inv_link(batch_theta_tildes)
	## forward model 
	batch_H_theta = sim_model.hrf_model(batch_thetas)
	batch_H_theta = batch_H_theta[...,start_m:,start_m:]
	## unrolled inference  
	signals_hat = net(batch_y_eps.to(device), batch_H_theta.to(device))
	loss = torch.mean((signals_hat - batch_neural_signals.to(device))**2)
	## update weights
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	#if epoch % num_epochs_track == 0:
	print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
	t2 = time.time() - t1
	if epoch % num_epochs_track == 0:
		with torch.no_grad():            
			test_signals_hat = net(y_eps_test.to(device), H_theta_test.to(device))
			test_loss = torch.mean((test_signals_hat - neural_signals_test.to(device))**2)
			# eval 
			print(f"Epoch {epoch}, Test Loss: {test_loss.item():.4f}")
			fname_net_state_dict = os.path.join(MODELDIR, "DUP_1param_%s.pth" % epoch)
			torch.save({
										"epoch": epoch,
										"test_losses":test_loss.item(),
										"model_state_dict": net.state_dict(),
										"optimizer_state_dict": optimizer.state_dict(),
										"iter_time":t2,
										}, fname_net_state_dict)
