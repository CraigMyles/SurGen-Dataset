<h1 align="center">
  SurGen: 1020 H&E-stained Whole Slide Images With Survival and Genetic Markers
</h1>


[Download Data](https://doi.org/10.6019/S-BIAD1285) | [ArXiv Preprint](https://arxiv.org/abs/2502.04946) | [Reproducibility](./reproducibility) | [Notebooks](./notebooks) | [Pre-encoded UNI embeddings](https://zenodo.org/records/14047723) | [Citation](#reference)

## Abstract:
<img src="https://github.com/user-attachments/assets/253ad08f-1e57-4121-a721-df3424f66eb5" width="200px" align="right" />
SurGen is a publicly available colorectal cancer dataset comprising 1,020 whole slide images (WSIs) from 843 cases. Each WSI is digitised at 40× magnification (0.1112 µm/pixel) and is accompanied by key genetic marker; <em>KRAS, NRAS, BRAF</em> as well as <em>mismatch repair (MMR) status</em> and <em>five-year survival</em> data (available for 426 cases). This repository provides standard train-validation-test splits and example scripts to facilitate machine learning experiments in computational pathology, biomarker discovery, and prognostic modelling. SurGen aims to advance cancer diagnostics by offering a consistent, high-quality dataset with rich annotations, supporting both targeted research on primary colorectal tumours and broader studies of metastatic sites.

---

<img src="https://github.com/user-attachments/assets/6ad0c3a8-dddc-44f2-bf2b-d2d702e6e5e6" width="100%" align="center" />

---

## Overview

SurGen is split into two sub-cohorts:
1. **SR386** – Primary colorectal cancer (427 WSIs) with five-year survival data.
2. **SR1482** – Colorectal cancer cases (593 WSIs) including metastatic lesions (liver, lung, peritoneum), with full biomarker data.

Each WSI is stored in Zeiss `.CZI` format. For convenience, precomputed patch embeddings (extracted using the UNI foundation model) are also available.

---

## Downloading the Dataset

The SurGen dataset is hosted on the **EBI FTP server**. You can download the Whole Slide Images (WSIs) for **both sub-cohorts (SR386 and SR1482)** using `wget`, an FTP client, or you can download directly from the EBI [website](https://doi.org/10.6019/S-BIAD1285).

### **Option 1: Using `wget`**
For most, the easiest way to download the WSIs is via `wget`:

#### **Download all SR386 WSIs:**
```bash
wget -r -np -nH --cut-dirs=6 ftp://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs/
```

#### **Download all SR1482 WSIs:**
```bash
wget -r -np -nH --cut-dirs=6 ftp://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR1482_WSIs/
```

This will download the respective data into `SR386_WSIs/` and `SR1482_WSIs/` folders in your current directory.
- `-np` no parent (prevents downloading higher-level directories).
- `-nH` no host (ignores 'ftp.ebi.ac.uk' in the local directory structure).
- `--cut-dirs=6` ensures you get a clean directory structure without extra nested folders.

---

### **Option 2: Using FTP**
If you prefer to use FTP, follow these steps:

1. Open a terminal and connect to the FTP server:
   ```bash
   ftp ftp.ebi.ac.uk
   ```
   - When prompted, enter `anonymous` as the username.
   - Press **Enter** for the password.

2. Navigate to the **SR386** directory:
   ```bash
   cd /biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs
   ```

   Or for the **SR1482** directory:
   ```bash
   cd /biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR1482_WSIs
   ```

3. Enable **binary mode** to correctly transfer `.CZI` files:
   ```bash
   binary
   ```

4. Download all `.CZI` files:
   ```bash
   prompt
   mget *.czi
   ```
5. Close the ftp connection:
   ```bash
   exit
   ```
   
---

### **Option 3: Using an FTP Client**
You can also use an FTP GUI client such as **FileZilla** or **Cyberduck**:

- **Host:** `ftp.ebi.ac.uk`
- **Username:** `anonymous`
- **Port:** `21`
- **Path:** `/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/`

---

## Reproducibility

The [reproducibility](./reproducibility) directory contains step-by-step instructions to replicate the results shown in our DataNote [paper](https://doi.org/10.48550/arXiv.2502.04946). These include:
- Details on environment setup and required dependencies.
- Scripts for processing the WSIs and generating patch-level features.
- Guidelines for reproducing slide-level prediction results.

This ensures that all experiments can be reliably reproduced by other researchers using the provided dataset and embeddings.

---

## Notebooks

The [notebooks](./notebooks) directory provides interactive examples for exploring the SurGen dataset and pre-extracted features:
1. **`simple_load_wsi_tile.ipynb`** – Demonstrates how to interact with `.CZI` files in Python, including reading and viewing from WSIs.  
2. **`patch_feature_extraction.ipynb`** – Shows how to extract patch-level features using Hugging Face models, this example uses the UNI foundation model.  
3. **`zarr_examined.ipynb`** – Explains the layout and usage of pre-extracted SurGen features stored in Zarr format, making it easier to integrate with downstream analysis pipelines.

These notebooks provide a practical starting point for using the dataset and applying it to various computational pathology tasks.

---

## Reference
If you find this dataset or repository useful, please consider citing the following:
```bibtex
@article{myles2025surgen,
  title={SurGen: 1020 H\&E-stained Whole Slide Images With Survival and Genetic Markers},
  author={Myles, Craig and Um, In Hwa and Marshall, Craig and Harris-Birtill, David and Harrison, David J},
  journal={arXiv preprint arXiv:2502.04946},
  year={2025}
}

@inproceedings{myles2024leveraging,
  title={Leveraging foundation models for enhanced detection of colorectal cancer biomarkers in small datasets},
  author={Myles, Craig and Um, In Hwa and Harrison, David J and Harris-Birtill, David},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={329--343},
  year={2024},
  organization={Springer}
}
```
