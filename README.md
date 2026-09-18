# Master Thesis — Radar-Based Precipitation Nowcasting over Belgium

This repository contains the code associated with my Master's thesis at the
Vrije Universiteit Brussel (VUB), carried out in the context of the MLCast
project.

The thesis investigates radar-based precipitation nowcasting over Belgium
using the BE RADCLIM 1 km composite precipitation dataset.

## Forecasting task

The neural models use four recent precipitation fields at 5-minute intervals
to forecast 20 future fields from +5 to +100 minutes.

The STEPS baseline implemented with PySTEPS uses an AR(2) configuration based
on the latest three precipitation fields.

Data from 2017--2020 and 2022--2023 are used for model development, while
2021 is kept as an independent test year.

## Models

The study evaluates:

- STEPS implemented with PySTEPS
- deterministic ConvGRU
- probabilistic ConvGRU ensemble
- deterministic U-Net
- probabilistic U-Net ensemble
- LDCast latent-diffusion nowcasting

Both a fixed 10% training subset and the full importance-sampled training
dataset are investigated.

## Repository structure

- `code/convgru/` — ConvGRU training and experiment code
- `code/unet/` — deterministic and probabilistic U-Net implementation
- `code/ldcast/` — LDCast training and experiment code
- `code/evaluation/` — 2021 evaluation and verification workflow
- `thesis/` — final thesis document
- `figures/` — selected thesis figures

## Main technical contribution

A main software contribution of this thesis is the development of
deterministic and probabilistic U-Net baselines for the MLCast workflow,
including Fiddle-based configuration, autoregressive multi-step nowcasting,
and probabilistic ensemble training.

## Evaluation

The final quantitative evaluation uses 1,460 cases from the independent
2021 test year.

Verification includes:

- pixel-wise error metrics
- categorical rainfall verification
- Fractions Skill Score (FSS)
- Continuous Ranked Probability Score (CRPS)
- Brier score and reliability analysis
- ensemble rank histograms

Verification is restricted to valid BE RADCLIM pixels inside the Belgian
national boundary.

## Data and large outputs

The BE RADCLIM dataset is not distributed in this repository.

Large model checkpoints, full inference archives, Zarr prediction stores,
and other generated outputs are excluded because of their size.

## Upstream projects

This work builds on existing open-source implementations and the MLCast
ecosystem:

- MLCast: https://github.com/mlcast-community/mlcast
- ConvGRU-Ensemble: https://github.com/DSIP-FBK/ConvGRU-Ensemble
- PySTEPS: https://github.com/pySTEPS/pysteps

The original licence files are retained with the corresponding source code.

## Author

Nhat Minh Hoang  
Vrije Universiteit Brussel  
Master's Thesis, 2026
