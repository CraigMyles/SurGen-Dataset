"""Constants for the datasets module."""

TASKS = {
  "SR386": [
    # None,
    "MMR_LOSS",
    "RAS_M", # RAS_M is a combination of KRAS_M, BRAF_M, NRAS_M but is it's own column
    "KRAS_M",
    "BRAF_M", # KRAS ex2, KRAS ex3, KRAS codon 117, KRAS codon 146
    "NRAS_M", # NRAS ex2, NRAS ex3
    "5Y_SUR", # 5 year survival (binary)
  ],
  "SR1482": [
    # None, 
    "MMR_MSI", 
    "RAS_M", 
    "KRAS_M", 
    "NRAS_M", 
    "BRAF_M"
  ],
  "SurGen": [
    # None, 
    "MMR_MSI"
  ]
}

AVAILABLE_COHORTS = list(TASKS.keys())

# label encoding for tasks
# for BRAF: 0 = WT, 1 = M
# for KRAS: 0 = WT, 1 = M
# for NRAS: 0 = WT, 1 = M
# MMR is already encoded
# 5Y_SUR is already encoded
# for RAS: 0 = WT, 1 = M

LABEL_ENCODINGS = {
  "SR386": {
    "MMR_LOSS": {"MMR Loss": 1, "No MMR Loss": 0},
    "RAS_M": {"WT": 0, "M": 1},
    "KRAS_M": {"WT": 0, "M": 1},
    "BRAF_M": {"WT": 0, "M": 1},
    "NRAS_M": {"WT": 0, "M": 1},
    "5Y_SUR": {"Died within 5y": 1, "Survived 5y": 0},
  },
  "SR1482": {
    "MMR_MSI": {"MSI/dMMR": 1, "MSS/pMMR": 0},
    "RAS_M": {"WT": 0, "M": 1},
    "KRAS_M": {"WT": 0, "M": 1},
    "BRAF_M": {"WT": 0, "M": 1},
    "NRAS_M": {"WT": 0, "M": 1},
  },
  "SurGen": {
    "MMR_MSI": {"MSI/dMMR": 1, "MSS/pMMR": 0},
  }
}