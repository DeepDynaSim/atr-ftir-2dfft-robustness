# ATR-FTIR 2D-FFT robustness analysis

This repository contains the MATLAB analysis and derived outputs for an independent patient-level evaluation of 2D-FFT feature engineering in ATR-FTIR COVID-19 severity classification.

The analysis uses the public Banerjee et al. plasma dataset (160 patients; 130 development and 30 held-out test patients) released at [Zenodo DOI 10.5281/zenodo.4805258](https://doi.org/10.5281/zenodo.4805258). The source data are not redistributed here.

## Main result

For the prespecified MLP ensemble, 1D-FFT magnitude + PCA and 2D-FFT 16×16 magnitude + PCA had the same held-out balanced accuracy (0.747). The paired difference for 2D-FFT minus 1D-FFT was 0.000 (95% stratified bootstrap interval -0.165 to 0.156). The AUROC difference was -0.023 (-0.113 to 0.057). Neither comparison supports an incremental benefit from the 2D embedding.

Changing the 2D matrix geometry, matching all representations to seven principal components, and retaining only the 130 unique coefficients from the Hermitian-symmetric 16×16 Fourier plane did not establish 2D-FFT superiority.

## Repository contents

- `code/validate_2dfft_external.m`: complete MATLAB analysis.
- `results/`: patient-level predictions, performance tables, bootstrap contrasts, repeated-validation results, MLP seed variability, DOI verification, figures, and the saved MATLAB workspace.
- `data/README.md`: source-data provenance, checksums, and the expected local directory structure.
- `.zenodo.json`: machine-readable metadata for the permanent Zenodo release.
- `RELEASE_NOTES.md`: versioned summary of the included analyses and outputs.

## Software

The analysis was run in MATLAB R2024b with Statistics and Machine Learning Toolbox and Deep Learning Toolbox. The script uses a fixed top-level random seed and explicitly seeded MLP members.

## Reproduction

1. Download the public source files from DOI `10.5281/zenodo.4805258`.
2. Extract the development and test archives under `data/zenodo_4805258/` using the directory structure shown in `data/README.md`.
3. Run `code/validate_2dfft_external.m` in MATLAB R2024b.

The script performs patient-level replicate averaging, interpolation to 256 points over 1800–900 cm⁻¹, SNV normalization, representation construction, development-only standardization and PCA, MLP ensembles, conventional classifier benchmarks, paired bootstrap analyses, matched-PC and nonredundant-2D sensitivity analyses, and 50×5-fold development validation.

## Fixed classifier settings

- MLP: 25 independently seeded `patternnet` members; one 10-node hidden layer; tanh hidden activation; softmax output; scaled conjugate-gradient training; 500 maximum epochs; minimum gradient `1e-6`; cross-entropy; regularization `0.1`; fixed probability threshold `0.5`.
- LDA: `fitcdiscr` with linear discriminant type and empirical class priors.
- Ridge logistic regression: `fitclinear`, logistic learner, ridge regularization, `lambda=1/n_development`, `lbfgs` solver.
- Linear SVM: `fitcsvm`, linear kernel, `BoxConstraint=1`, `Standardize=false`, empirical priors, followed by `fitPosterior` on development data.
- RBF SVM: `fitcsvm`, RBF kernel, `KernelScale='auto'`, `BoxConstraint=1`, `Standardize=false`, empirical priors, followed by `fitPosterior` on development data.

No classifier or hyperparameter was selected using held-out performance.

## Licenses

The analysis code is released under the MIT License. Derived tables and figures are released under CC BY 4.0. The source spectra remain subject to the license and attribution requirements of the original Zenodo record.

## Citation

See `CITATION.cff`. Cite the archived release as [Zenodo DOI 10.5281/zenodo.23080999](https://doi.org/10.5281/zenodo.23080999); the development repository is [DeepDynaSim/atr-ftir-2dfft-robustness](https://github.com/DeepDynaSim/atr-ftir-2dfft-robustness).
