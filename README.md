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
- code for generating synthetic datasets used in the simulation study
- documentation on pulling the HCP-YA data to reproduce the real data analysis (not shipped due to size)

Most scripts use relative paths assuming they are run from the directory they live in.  
For example:

- run `SimStudy/posterior_inverter_1param.py` from `SimStudy/`
- run `RealDataAnalysis/population/hcp_1param_inf.py` from `RealDataAnalysis/population/`

---

## Download HCP data 

For all resting-state fMRI analyses, we use HCP surface-mapped 32k-vertex resting-state BOLD data after ICA-FIX preprocessing. In the local folders used by this code, the resting-state data are stored as plain-text `.1D` files with shape `32492 x 1200` (`vertices x timepoints`), and the cortical surfaces are stored as `.ply` meshes converted from the corresponding HCP 32k midthickness surfaces.

### General HCP files to download

The HCP-YA data can be downloaded at https://balsa.wustl.edu/. For a given subject `SUBJ`, the relevant HCP-YA files are:

- `SUBJ/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii` 
- `SUBJ/MNINonLinear/fsaverage_LR32k/SUBJ.L.midthickness_MSMAll.32k_fs_LR.surf.gii`

in `SUBJ_Rest3TRecommended` and `SUBJ_StructuralRecommended`, respectively. Any conversion script is fine as long as the final local files have the names expected by the scripts. In particular:

- the `.1D` files should contain the left cortical time series only, with `32492` rows and `1200` columns
- the surface files should be converted to `.ply`

### Calibration

To calibrate the simulation hyperparameters, download HCP-YA subject `103818` and extract the left-hemisphere `REST1_LR` surface time series. Place it at:

- `Calibration/Data_103818/103818_1_LR.L.1D`

### SimStudy

Download the rsHRF package (github.com/compneuro-da/rsHRF) and place it in the `SimStudy/CompetingMethods` subfolder. Create the subdirectories `SimStudy/data`, `SimStudy/UQ`, and `SimStudy/UQ/2param_bootsamples` which will hold results.

### RealDataAnalysis/reproducibility

Download the baseline and retest `REST1_LR` left-hemisphere data for each of the subjects

- `103818`
- `122317`
- `139839`

and convert the corresponding left midthickness surfaces to `.ply.` Place the baseline-visit files at:

- `RealDataAnalysis/reproducibility/HCP_retest/SUBJ_1_LR.L.1D`
- `RealDataAnalysis/reproducibility/HCP_retest/SUBJ.L.midthickness.32k_fs_LR.ply`

Place the retest-visit files at:

- `RealDataAnalysis/reproducibility/HCP_retest/Retest/SUBJ_1_LR.L.1D`
- `RealDataAnalysis/reproducibility/HCP_retest/Retest/SUBJ.L.midthickness.32k_fs_LR.ply`

where `SUBJ` is one of `103818`, `122317`, or `139839`.

Create the subdirectory `RealDataAnalysis/reproducibility/results`, which will hold the output of 
`hcp_restest_1param_inv.py`.

### RealDataAnalysis/population

The population analysis uses the 100 HCP-YA unrelated subjects listed in:

- `RealDataAnalysis/population/hcp_ya_unrelated_100_subjects.txt`

For each subject `SUBJ` in that list, download:

- `SUBJ/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii`
- `SUBJ/MNINonLinear/fsaverage_LR32k/SUBJ.L.midthickness_MSMAll.32k_fs_LR.surf.gii`

Then convert/place the outputs into:

- `RealDataAnalysis/population/S1200_Rest3TRecommended/SUBJ.L.1D`
- `RealDataAnalysis/population/S1200_StructuralRecommended/SUBJ.L.midthickness_MSMAll.32k_fs_LR.ply`

Create the subdirectories `RealDataAnalysis/population/S1200_RestT3_Estimates` and `RealDataAnalysis/population/fpca`, which will hold results.

### RealDataAnalysis/connectivity

The connectivity case study uses HCP-YA subject `103818` from the baseline visit. Download the HCP-YA subject `103818` and extract the left-hemisphere `REST1_LR` surface time series, and the subject-specific left midthickness surface. Place them at:

- `RealDataAnalysis/connectivity/data/103818_1_LR.L.1D`
- `RealDataAnalysis/connectivity/data/103818.L.midthickness.32k_fs_LR.ply`

## Code Details

### 1. `Calibration/`

Estimate simulator hyperparameters using the spectral moment-matching approach from Section **3.4.2**

| Path | Description |
| --- | --- |
| `Calibration/calibrate_1parambasis.ipynb` | Calibrates simulator hyperparameters for the one-parameter model 
| `Calibration/calibrate_2parambasis.ipynb` | Calibrates simulator hyperparameters for the two-parameter model. 
| `Calibration/Data_103818/` | rsfMRI data for HCP subject `103818`, used to calibrate simulators. 

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
| `SimStudy/posterior_inverter_1param.py` | Synthetic inversion script for the one-parameter hemodynamic model. | 
| `SimStudy/UQ_1param.py` | Runs the one-parameter double-bootstrap UQ procedure on synthetic data. WARNING: TAKES A LONG TIME, SHOULD DISTRIBUTE AND RUN ON CLUSTER. |
| `SimStudy/posterior_inverter_2param.py` | Synthetic inversion script for the two-parameter hemodynamic model. |
| `SimStudy/UQ_2param.py` | Runs the two-parameter double-bootstrap UQ procedure on synthetic data. WARNING: TAKES A LONG TIME, SHOULD DISTRIBUTE AND RUN ON CLUSTER. | 

#### Competing methods

| Path | Description | 
| --- | --- | 
| `SimStudy/CompetingMethods/bcd_eval.py` | Evaluates the joint-MAP/alternating-optimization baseline for the one-parameter synthetic study. |
| `SimStudy/CompetingMethods/unrolled.py` | Trains the deep unrolled prior baseline (DUP) for the one-parameter synthetic study. |
| `SimStudy/CompetingMethods/unrolled_eval.py` | Evaluates the trained DUP model on the one-parameter synthetic study. |
| `SimStudy/CompetingMethods/rsHRF_eval_sim.m` | MATLAB code for the `rsHRF` competitor used in the two-parameter synthetic study. | 

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
| `RealDataAnalysis/population/hcp_ya_unrelated_100_subjects.txt` | Subject IDs for the 100-subject HCP-YA unrelated cohort used in the population study. |
| `RealDataAnalysis/population/S1200_templates/` | Template surfaces used for visualization and fPCA. | 

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
| `RealDataAnalysis/connectivity/hemo_inf/` | Precomputed hemodynamic reconstructions used as inputs to the connectivity analysis. |
| `RealDataAnalysis/connectivity/atlas/` | Glasser atlas labels and lookup tables. | 

---

## Minimal Reproduction Workflow

**First, tollow all steps in Section `Download HCP data` to pull the data and set up proper directory structure.**

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
