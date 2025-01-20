<h1 align="center">
  SurGen: 1020 H&E-stained Whole Slide Images With Survival and Genetic Markers
</h1>


[Download Data](https://doi.org/10.6019/S-BIAD1285) | [Reproducibility](./reproducibility) | [Notebooks](./notebooks) | [Pre-encoded UNI embeddings](https://zenodo.org/records/14047723) | [GitHub Repository](https://github.com/CraigMyles/SurGen-Dataset) | [Citation](#reference)



## Abstract:
SurGen is a publicly available colorectal cancer dataset comprising 1,020 whole slide images (WSIs) from 843 cases. Each WSI is digitised at 40× magnification (0.1112 µm/pixel) and is accompanied by key genetic marker; <em>KRAS, NRAS, BRAF</em> as well as <em>mismatch repair (MMR) status</em> and <em>five-year survival</em> data (available for 426 cases). This repository provides standard train-validation-test splits and example scripts to facilitate machine learning experiments in computational pathology, biomarker discovery, and prognostic modelling. SurGen aims to advance cancer diagnostics by offering a consistent, high-quality dataset with rich annotations, supporting both targeted research on primary colorectal tumours and broader studies of metastatic sites.

<img src="https://github.com/user-attachments/assets/4724e007-96f3-4172-bdbc-b00324966300" width="100%" align="center" />

---

## Overview

SurGen is split into two sub-cohorts:
1. **SR386** – Primary colorectal cancer (427 WSIs) with five-year survival data  
2. **SR1482** – Colorectal cancer cases (593 WSIs) including metastatic lesions (liver, lung, peritoneum), with full biomarker data

Each WSI is stored in `.CZI` format. For convenience, precomputed patch embeddings (extracted using the UNI foundation model) are also available. Refer to the [GitHub Repository](https://github.com/CraigMyles/SurGen-Dataset) for usage examples and data-processing scripts.

## Reproducibility

The `reproducibility` directory contains step-by-step instructions to replicate the results shown in the associated paper. These include:
- Details on environment setup and required dependencies.
- Scripts for processing the WSIs and generating patch-level features.
- Guidelines for reproducing survival analysis and biomarker prediction results.

This ensures that all experiments can be reliably reproduced by other researchers using the provided dataset and embeddings.

## Notebooks

The `notebooks` directory provides interactive examples for exploring the SurGen dataset and pre-extracted features:
1. **`simple_load_wsi_tile.ipynb`** – Demonstrates how to interact with `.CZI` files in Python, including reading and viewing from WSIs.  
2. **`patch_feature_extraction.ipynb`** – Shows how to extract patch-level features using Hugging Face models, leveraging the UNI foundation model.  
3. **`zarr_examined.ipynb`** – Explains the layout and usage of pre-extracted SurGen features stored in Zarr format, making it easier to integrate with downstream analysis pipelines.

These notebooks provide a practical starting point for using the dataset and applying it to various computational pathology tasks.

## Reference
When referencing this work, please use the following:
```bibtex
@article{myles2025surgen,
  author = {Craig Myles and In Hwa Um and Craig Marshall and David Harris-Birtill and David J Harrison},
  title = {SurGen: 1020 H&E-stained Whole Slide Images With Survival and Genetic Markers},
  year = {2025},
  note = {Manuscript under review.}
}
```
