# Source data

The analysis uses the public dataset:

Banerjee A, Gokhale A, Bankar R, et al. Rapid classification of COVID-19 severity by ATR-FTIR spectroscopy of plasma samples. Zenodo. 2021. DOI: `10.5281/zenodo.4805258`.

The source record is licensed under CC BY 4.0. Source archives are not redistributed in this repository.

## Verified source files

| File | MD5 |
| --- | --- |
| Data for predictive model.rar | `f59458452c98deaebbac93c293fdb681` |
| Data for testing the model.rar | `549500140aed3d2048604ae42925b3b1` |
| Supporting information for publication | `81a086b265ed26e2e77587483d6e89e3` |

## Expected extracted structure

```text
data/zenodo_4805258/
  predictive_extracted/
    Data for predictive model/
      Non Severe/CSV Format/*.csv
      Severe/CSV Format/*.csv
  testing_extracted/
    Data for testing the model/
      Blinded Study 10112020/
        CSV Format/*.csv
```

The MATLAB script verifies 130 development patients, 30 held-out test patients, the class counts, and exactly two technical replicates per patient before model fitting.
