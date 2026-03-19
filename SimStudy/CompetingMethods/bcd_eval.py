import numpy as np
import torch
import torch.nn as nn
from torch.special import gammaln, xlogy
from torch.utils.data import DataLoader, Dataset

import scipy
from scipy.fft import fft 
from torch.distributions.multivariate_normal import MultivariateNormal
import cvxpy as cp

import os 
from scipy.io import loadmat
import time 

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


#### BCD Algorithm ####
def update_s(y, H, lam):
	s = cp.Variable(H.shape[0], nonneg=True)
	obj = 0.5*cp.sum_squares(H @ s - y) + lam*cp.norm1(s)
	prob = cp.Problem(cp.Minimize(obj))
	prob.solve(solver=cp.OSQP, warm_start=True)
	return torch.from_numpy(s.value).float()

def update_theta(y, s):
	def f_theta(th):
		hrf_kernel_theta, _ = hrf_model._double_gamma_hrf(th)
		H_theta = hrf_model._construct_toeplitz(hrf_kernel_theta)
		return 0.5*torch.sum((H_theta @ s - y)**2).item()
	return golden_section_minimize(f_theta, t1_min, t1_max)

def golden_section_minimize(f, a, b, tol=1e-3, maxit=50):
	phi = (1 + np.sqrt(5)) / 2
	invphi = 1/phi
	invphi2 = (3 - np.sqrt(5)) / 2
	c = b - invphi*(b - a)
	d = a + invphi*(b - a)
	fc, fd = f(torch.tensor([c])), f(torch.tensor([d]))
	it = 0
	while (b - a) > tol and it < maxit:
		if fc < fd:
			b, d, fd = d, c, fc
			c = b - invphi*(b - a)
			fc = f(torch.tensor([c]))
		else:
			a, c, fc = c, d, fd
			d = a + invphi*(b - a)
			fd = f(torch.tensor([d]))
		it += 1
	theta = (a + b) / 2
	return theta

def BCD(y_obs, lam, max_iter=100, tol=1e-3):
	## initialize parameters
	theta_traj = torch.zeros(max_iter)
	s_traj = torch.zeros(M, max_iter)
	theta_traj[0] = torch.tensor([0.5*(t1_min + t1_max)])
	for i in range(1,max_iter):
		## update s 
		hrf_kernel_theta, _ = hrf_model._double_gamma_hrf(theta_traj[i-1])
		H_theta = hrf_model._construct_toeplitz(hrf_kernel_theta)
		s_traj[:,i] = update_s(y_obs, H_theta, lam)
		## update theta 
		theta_traj[i] = update_theta(y_obs, s_traj[:,i])
		## assess convergence 
		if np.abs(theta_traj[i] - theta_traj[i-1]) < tol:
			break 
	return theta_traj[:i], s_traj[:,:i]

## HCP-parameters 
M = 1200
t_r =  0.72 
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

#### Load Simulated Data #### 
simfname = "../data/sim1data_surface.mat"
sim1data = loadmat(simfname)
y_obs = torch.from_numpy(sim1data["y_obs"]).float()
y_true = torch.from_numpy(sim1data["y_true"]).float()
neural_signals_true = torch.from_numpy(sim1data["neural_signals_true"]).float()
theta_true = torch.from_numpy(sim1data["theta_true"]).float()
theta_tilde_true = torch.from_numpy(sim1data["theta_tilde_true"]).float()
V = y_obs.shape[0]

## regularization paramter 
reg_param = np.sqrt(np.log(M)*2)*sigma_e ## universal threshold
theta_hat = torch.zeros(V,1)
for v in range(V):
	y_v = y_obs[v,:]
	theta_traj_v, s_traj_v = BCD(y_v, reg_param)
	theta_hat[v] = theta_traj_v[-1]
	if not (v % 500):
		print(v)

print("BCD Inference Results")
print("L2 Error", torch.mean((theta_hat - theta_true)**2))
print("Normalized L2 Error", torch.norm(theta_hat - theta_true)/torch.norm(theta_true).item())
print("Bias", torch.mean(theta_hat - theta_true))

