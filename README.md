<h1 align="center">
  SurGen: 1020 H&E-stained Whole Slide Images With Survival and Genetic Markers
</h1>


[Download Data](https://doi.org/10.6019/S-BIAD1285) | [Pre-encoded UNI embeddings](https://zenodo.org/records/14047723) | [GitHub Repository](https://github.com/CraigMyles/SurGen-Dataset) | [Citation](#reference)



## Abstract:
SurGen is a publicly available colorectal cancer dataset comprising 1,020 whole slide images (WSIs) from 843 cases. Each WSI is digitised at 40× magnification (0.1112 µm/pixel) and is accompanied by key genetic markers—<em>KRAS, NRAS, BRAF</em>—as well as mismatch repair (MMR) status and five-year survival data (available for 426 cases). This repository provides standard train-validation-test splits and example scripts to facilitate machine learning experiments in computational pathology, biomarker discovery, and prognostic modelling. SurGen aims to advance cancer diagnostics by offering a consistent, high-quality dataset with rich annotations, supporting both targeted research on primary colorectal tumours and broader studies of metastatic sites.

<img src="https://github.com/user-attachments/assets/4724e007-96f3-4172-bdbc-b00324966300" width="100%" align="center" />

---

## Overview

SurGen is split into two sub-cohorts:
1. **SR386** – Primary colorectal cancer (427 WSIs) with five-year survival data  
2. **SR1482** – Colorectal cancer cases (593 WSIs) including metastatic lesions (liver, lung, peritoneum), with full biomarker data

Each WSI is stored in `.CZI` format. For convenience, precomputed patch embeddings (extracted using the UNI foundation model) are also available. Refer to the [GitHub Repository](https://github.com/CraigMyles/SurGen-Dataset) for usage examples and data-processing scripts.

---

## Reference
When referencing this work, please use the following:
```bibtex
tbc
```
