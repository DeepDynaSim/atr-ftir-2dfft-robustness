# Version 1.0.0

Initial reproducibility release for the independent ATR-FTIR COVID-19 severity representation study.

## Included analyses

- Direct paired 10,000-resample bootstrap comparison of 2D-FFT 16×16 + PCA versus 1D-FFT magnitude + PCA for balanced accuracy and AUROC.
- Fifty repetitions of stratified five-fold development validation summarized by median and IQR.
- Fully specified MLP, LDA, ridge-logistic, linear-SVM, and RBF-SVM settings.
- Individual results for 25 MLP seeds.
- Seven-PC matched-dimensionality sensitivity analysis.
- Nonredundant Hermitian half-plane 2D-FFT sensitivity analysis.
- Patient-level predictions, conventional-classifier benchmarks, figures, and the saved MATLAB workspace.
- Crossref/DataCite verification of all 22 manuscript DOIs.

## Source data

The public plasma spectra are available from DOI `10.5281/zenodo.4805258` and are not redistributed in this release.
