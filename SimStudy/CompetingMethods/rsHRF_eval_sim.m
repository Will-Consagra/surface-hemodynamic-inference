%% Load Simulated Data
clc,clear;close all;

%load neural_signals_true, theta_true, y_obs, y_true    
load ../../SimStudy/data/sim2data_surface.mat
t_r = 0.72; 
sigma_e = 0.24212532107069681;
V = size(theta_true,1);
M = size(y_obs,2);

%% rsHRF Estimation
addpath(genpath('rsHRF'))


nobs=M;  
TR = t_r;

%%===========PARAMETERS========================
para.TR = TR;
BF = {'Canonical HRF (with time derivative)'
'Canonical HRF (with time and dispersion derivatives)'              
'Gamma functions'
'Fourier set'
'Fourier set (Hanning)'};
% choose the set of basis functions THIS MUST BE AN INPUT
bf_id = 1;
para.name = BF{bf_id}; 
para.order = 1; 

% no mask
temporal_mask = []; 

para.T  = 1; % magnification factor of temporal grid with respect to TR. i.e. para.T=1 for no upsampling, para.T=3 for 3x finer grid
para.T0 = 1; % position of the reference slice in bins, on the grid defined by para.T. For example, if the reference slice is the middle one, then para.T0=fix(para.T/2)
if para.T==1
    para.T0 = 1;
end

para.dt  = para.TR/para.T; % fine scale time resolution.
para.AR_lag = 0; % AR(0) no noise autocorrelation
para.thr = 1; % (mean+) para.thr*standard deviation threshold to detect event.

para.len = 29; % length of HRF, in seconds (match to L param in posterior inverter)

min_onset_search = 4; % minimum delay allowed between event and HRF onset (seconds)
max_onset_search = 8; % maximum delay allowed between event and HRF onset (seconds)
para.lag  = fix(min_onset_search/para.dt):fix(max_onset_search/para.dt);

%%=============HRF estimation======================

rsHRF_estim_params = zeros(V, 2); 
rsHRF_estim_qoi = zeros(V, 3); %response hieght, time to peak, FWHM
rsHRF_estim_kernel = zeros(V, 41);

%bandpass filter lower and upper bound
bands=[0.01 0.1]; 

for v=1:V
    bold_sig = y_obs(v,:)';
    %apply bandpass filter
    bands=[0.01 0.1]; 
    bold_sig = bold_sig-mean(bold_sig);
    data = rsHRF_band_filter(bold_sig,TR,bands);
    sigma = std(data);
    %tic
    [beta_hrf, bf, event_bold] = rsHRF_estimation_temporal_basis(data,para,temporal_mask);
    hrfa = bf*beta_hrf(1:size(bf,2),:); %HRF
    nvar = size(hrfa,2); PARA = zeros(3,nvar);
    for voxel_id=1:nvar
        hrf1 = hrfa(:,voxel_id);
        PARA(:,voxel_id) = rsHRF_get_HRF_parameters(hrf1,para.dt); % estimate HRF parameter
    end
    rsHRF_estim_params(v,:) = beta_hrf(1:size(bf,2),:);
    rsHRF_estim_qoi(v,:) = PARA;
    rsHRF_estim_kernel(v,:) = hrf1;
end 

%% Evaluation 
%figure(1);
%plot((1:length(hrf_estims(1,:)))*TR/para.T,hrf_estims(1,:),'b');
%hold on
%plot((1:length(hrf_estims(2,:)))*TR/para.T,hrf_estims(2,:),'g');

%figure(1);
%plot((1:length(hrfa(:,1)))*TR/para.T,hrfa(:,1),'b');
%xlabel('Time (s)')
%title(['HRF (',BF{bf_id},')'])
%hold on;
%plot((1:length(hrf(:,1)))*TR/para.T,hrf(:,1),'r');

load ../data/sim2data_inference_proposed.mat


% kernel parameters errors 
% bound estimates for rsHRF to parameter range for fair comparison 
t1_min = 0.5; t1_max = 1.5;
t2_min = -1.0; t2_max = 1.0;

lower = [t1_min, t2_min];
upper = [t1_max, t2_max];

rsHRF_estim_params_clipped = max(min(rsHRF_estim_params, upper), lower);

% kernel parameters (nonlinear inversion)
% L2 error 
rsHRF_param_errors = mean((rsHRF_estim_params_clipped - theta_true).^2);
rsHRF_param_errors


% Bias 
rsHRF_param_bias = mean(rsHRF_estim_params_clipped - theta_true);
rsHRF_param_bias


