# Learned Hemodynamic Coupling Inference in Resting-State Functional MRI

This repository accompanies the paper  
**“Learned Hemodynamic Coupling Inference in Resting-State Functional MRI”**  

[arXiv:2601.00973](https://arxiv.org/abs/2601.00973)

## High-Level Description

This code implements the workflow described in Sections 2–5 of the paper:

- simulator construction for latent neural activity and hemodynamics
- learned marginal-likelihood training via a summary network and conditional normalizing flow
- reproducible synthetic-data experiments
- in-vivo HCP analyses on the cortical surface:
  - test–retest reproducibility
  - population-level analysis
  - effective connectivity analysis

The repo also ships:
- pre-trained neural networks
- synthetic datasets used in the simulation study
- HCP cortical surface rs-fMRI data used in the real-data analyses

Most scripts use relative paths assuming they are run from the directory they live in.  
For example:

- run `SimStudy/posterior_inverter_1param.py` from `SimStudy/`
- run `RealDataAnalysis/population/hcp_1param_inf.py` from `RealDataAnalysis/population/`

---

## Repository Details

### 1. `Calibration/`

Estimate simulator hyperparameters using the spectral moment-matching approach from Section **3.4.2**

| Path | Description |
| --- | --- |
| `Calibration/calibrate_1parambasis.ipynb` | Calibrates simulator hyperparameters for the one-parameter model 
| `Calibration/calibrate_2parambasis.ipynb` | Calibrates simulator hyperparameters for the two-parameter model. 
| `Calibration/Data_103818/` | Surface mesh and rsfMRI files for HCP subject `103818`, used as to calibrate. 

### 2. `ModelTraining/`

Train the summary networks and likelihood emulators described in Section **3.1**

| Path | Description |
| --- | --- | 
| `ModelTraining/train_summary_network_1param.py` | Trains the summary network for the one-parameter model |
| `ModelTraining/train_lik_emul_1param.py` | Trains the one-parameter conditional normalizing-flow emulator |
| `ModelTraining/train_summary_network_2param.py` | Trains the summary network for the two-parameter model |
| `ModelTraining/train_likelihood_emulator_2param.py` | Trains the two-parameter conditional normalizing-flow emulator |

### 3. `PreTrainedModels/`

These are the pre-trained models that are used by the analysis scripts. They correspond to the models trained as described in Section **4.3**.

### 4. `SimStudy/`

Reproduce experimental results from Section **5.1**. Most of the scripts print results to screen that can then be assembled to form the results in Table 1.

#### Main scripts

| Path | Description |
| --- | --- | 
| `SimStudy/posterior_inverter_1param.py` | Synthetic inversion script for the one-parameter model. | 
| `SimStudy/UQ_1param.py` | Runs the one-parameter double-bootstrap UQ procedure on synthetic data. WARNING: TAKES A LONG TIME, SHOULD DISTRIBUTE AND RUN ON CLUSTER. |
| `SimStudy/posterior_inverter_2param.py` | Main synthetic inversion script for the two-parameter basis model. |
| `SimStudy/UQ_2param.py` | Runs the two-parameter double-bootstrap UQ procedure on synthetic data. WARNING: TAKES A LONG TIME, SHOULD DISTRIBUTE AND RUN ON CLUSTER. | 

#### Competing methods

| Path | Description | 
| --- | --- | 
| `SimStudy/CompetingMethods/bcd_eval.py` | Evaluates the joint-MAP/alternating-optimization baseline for the one-parameter synthetic study. |
| `SimStudy/CompetingMethods/unrolled.py` | Trains the deep unrolled prior baseline (DUP) for the one-parameter synthetic study. |
| `SimStudy/CompetingMethods/unrolled_eval.py` | Evaluates the trained DUP model on the one-parameter synthetic study. |
| `SimStudy/CompetingMethods/rsHRF_eval_sim.m` | MATLAB code for the `rsHRF` competitor used in the two-parameter synthetic study. | 
| `SimStudy/CompetingMethods/rsHRF/` | Copied MATLAB package used by `rsHRF_eval_sim.m`. |

#### Data

| Path | Description | 
| --- | --- | 
| `SimStudy/cortical_mesh/midthickness.32k_fs_LR.ply` | Left-hemisphere cortical mesh used for synthetic surface experiments. |
| `SimStudy/data/sim1data_surface.mat` | Synthetic one-parameter rsfMRI dataset used in the simulation study. | 
| `SimStudy/data/sim2data_surface.mat` | Synthetic two-parameter rsfMRI dataset used in the simulation study. | 
| `SimStudy/data/h_t.pth` | Canonical HRF basis function used in the two-parameter model. | 
| `SimStudy/data/dh_t.pth` | Time-derivative HRF basis used in the two-parameter model. | 
| `SimStudy/UQ/` | Holds bootstrap samples and other outputs from the UQ scripts. | 

### 5. `RealDataAnalysis`

Reproduce experimental results from Section **5.2**. 

#### Population

HCP population analysis in Section **5.2.2**. 

| Path | Description |  
| --- | --- |
| `RealDataAnalysis/population/hcp_1param_inf.py` | Runs one-parameter inversions on the HCP-YA unrelated cohort and writes out per-subject hemodynamic fields.| 
| `RealDataAnalysis/population/hcp_1param_analysis.py` | Performs downstream population analysis on the reconstructed subject-level fields: mean field, variability, fPCA, and proof-of-concept regressions on subject-level outcomes. |
| `RealDataAnalysis/population/SMs.csv` | HCP subject-measure table used in the population regression analyses. | 
| `RealDataAnalysis/population/conf.csv` | Confound/covariate table used to adjust the population regressions. | 
| `RealDataAnalysis/population/S1200_Rest3TRecommended/` | Input rsfMRI time series for the 100 unrelated HCP-YA subjects used in the population study. NOT SHIPPED WITH CODE DUE TO SIZE. WILL NEED TO DOWNLOAD THESE FROM PUBLICLY AVAILABLE HCP REPO! |
| `RealDataAnalysis/population/S1200_StructuralRecommended/` | Subject-specific cortical surface meshes corresponding to the population study inputs. | 
| `RealDataAnalysis/population/S1200_templates/` | Template surfaces used for visualization and fPCA summaries. | 
| `RealDataAnalysis/population/S1200_RestT3_Estimates/` | Output directory populated by `hcp_1param_inf.py` with subject-level hemodynamic estimates. | 
| `RealDataAnalysis/population/fpca/` | Output directory populated by `hcp_1param_analysis.py` containing the mean field, standard deviation, and eigenfunctions. |

#### Reproducibility

HCP test–retest analysis in Section **5.2.1**. 

| Path | Description | 
| --- | --- | 
| `RealDataAnalysis/reproducibility/hcp_restest_1param_inv.py` | Runs one-parameter inversion and interval estimation on the HCP test–retest dataset. Used for Figure 3. |
| `RealDataAnalysis/reproducibility/hcp_retest_1param_inv_hpselect.py` | Used for hyperparameter selection. |
| `RealDataAnalysis/reproducibility/HCP_retest/` | Input HCP test–retest surfaces and rsfMRI for the reproducibility analysis. | 
| `RealDataAnalysis/reproducibility/results/` | Output directory containing retest reconstructions and derived summaries. | 

#### Connectivity 

This directory contains the connectivity analysis from Section **5.2.3**.

| Path | Description | 
| --- | --- | 
| `RealDataAnalysis/connectivity/connectivity_analysis_1param.py` | Compares effective connectivity using raw BOLD, canonical-HRF deconvolution, and deconvolution under the estimated spatially varying hemodynamics. |
| `RealDataAnalysis/connectivity/data/` | Subject-level input BOLD/surface files used in the connectivity case study. | 
| `RealDataAnalysis/connectivity/hemo_inf/` | Precomputed hemodynamic reconstructions used as inputs to the connectivity analysis. |
| `RealDataAnalysis/connectivity/atlas/` | Glasser atlas labels and lookup tables. | 
| `RealDataAnalysis/connectivity/results/` | Output directory for connectivity figures and parcel summaries (Figure 5). | 

---

## Minimal Reproduction Workflow

### To reproduce the synthetic experiments
1. Use the pretrained models from `PreTrainedModels/` 
2. Run `SimStudy/posterior_inverter_1param.py`
3. Run `SimStudy/UQ_1param.py` (distribute on cluster or this will take a long time)
4. Run `SimStudy/posterior_inverter_2param.py`
5. Run `SimStudy/UQ_2param.py` (distribute on cluster or this will take a long time)
6. Run baselines under `SimStudy/CompetingMethods/`

### To reproduce the in-vivo analyses
1. Reconstruct subject-level fields with `RealDataAnalysis/population/hcp_1param_inf.py`
2. Run:
   - `RealDataAnalysis/reproducibility/hcp_restest_1param_inv.py` for Section 5.2.1
   - `RealDataAnalysis/population/hcp_1param_analysis.py` for Section 5.2.2
   - `RealDataAnalysis/connectivity/connectivity_analysis_1param.py` for Section 5.2.3


