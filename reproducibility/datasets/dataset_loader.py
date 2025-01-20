from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional
import zarr
import torch
import math
import pandas as pd

from datasets.dataset_constants import LABEL_ENCODINGS
import logging

def print_gpu_memory_stats(tag=""):
    if torch.cuda.is_available():
        logging.debug(f"{tag} - GPU Memory Usage")
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        cached = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert bytes to GB
        logging.debug(f"Allocated memory: {allocated:.4f} GB")
        logging.debug(f"Cached memory: {cached:.4f} GB")
    else:
        logging.debug("CUDA is not available. No GPU detected.")

# This class handles entire datasets
class Dataset_All_Bags(Dataset):
    def __init__(self, csv_path: Path, cohort: str, task: str, shuffle: bool = False):
        print_gpu_memory_stats("Before loading CSV")
        self.df = pd.read_csv(csv_path)
        print_gpu_memory_stats("After loading CSV")
        self.task = task
        self.cohort = cohort

        # Ensure LABEL_ENCODINGS contains the necessary cohort and task information
        if cohort not in LABEL_ENCODINGS or task not in LABEL_ENCODINGS[cohort]:
            raise ValueError(f"No label encodings found for cohort '{cohort}' and task '{task}'")

        self.label_encodings = LABEL_ENCODINGS[cohort][task]

        # Translate Descriptive Labels to Numeric Values
        self.translate_descriptive_labels()

        # Validate Numeric Labels
        self.validate_labels()

        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            print_gpu_memory_stats("After shuffling")

    def translate_descriptive_labels(self):
        # Translate each descriptive label into its numeric value based on LABEL_ENCODINGS
        for label_key, label_value in self.label_encodings.items():
            self.df['label'] = self.df['label'].replace(label_key, label_value)

    def validate_labels(self):
        # Convert labels to numeric type to ensure proper comparison
        self.df['label'] = pd.to_numeric(self.df['label'], errors='coerce')

        # Identify any rows with labels that are NaN (due to invalid conversion) or not in the expected range
        invalid_label_mask = self.df['label'].isna() | ~self.df['label'].isin(self.label_encodings.values())
        if invalid_label_mask.any():
            invalid_labels = self.df[invalid_label_mask]
            print(f"Warning: Invalid labels found:\n{invalid_labels}")

            # Remove rows with invalid labels
            self.df = self.df[~invalid_label_mask]
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Ensure 'case_id' and 'label' columns are present in the dataframe
        if 'case_id' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError("DataFrame does not contain 'case_id' or 'label' columns")

        case_id = self.df.iloc[idx]['case_id']
        label = self.df.iloc[idx]['label']

        return case_id, label

        

# Class to handle zarr data (sets of WSI feature vectors. Each WSI has its own zarr file). 
    # This class is responsible for loading feature vectors and coordinates for individual WSI instances
class SlideDataset(Dataset):
    def __init__(
        self,
        slide: Union[str, Path],
        cohort: str,
        task: str,
        batch_size: Optional[int] = None,
    ):

        print_gpu_memory_stats("Before loading zarr")
        slide = Path(slide)
        # self.label_encodings = LABEL_ENCODINGS[cohort][task]
        # print(f"self.label_encodings: {self.label_encodings}")
        #storage format determined by file extension found in wsi_vector_path
        self.storage_format = slide.suffix
        if self.storage_format not in ['.zarr']:
            raise ValueError("Invalid storage_format: implemented only for zarr at present.")
        self.zarr_group = zarr.open_group(str(slide), mode="r")
        print_gpu_memory_stats("After loading zarr")
        self.batch_size = batch_size
        self.num_patches = self.zarr_group["features"].shape[0]
        self.num_batches = math.ceil(self.num_patches / self.batch_size) if self.batch_size else 1

        # self.label = self.zarr_group.attrs["label"] # sometimes 1,0 sometimes M,WT. 
        #get labels from the 'label' column in the csv file
        

        self.name = slide.stem
        # self.transform = transform
        # self.inverse_transform = inverse_transform

    def __len__(self):
        # print(f"in len method, num_batches = {(self.num_batches)}")
        return self.num_batches

    def __getitem__(self, idx):
        print_gpu_memory_stats("Before retrieving features")
        features = self.zarr_group["features"][:]
        assert features.shape[0] == self.num_patches, f"Expected {self.num_patches} patches, got {features.shape[0]}"
        # coords = self.zarr_group["coords"][:]
        #coords is not currently needed. Return empty array which is same shape as features
        coords = torch.zeros(features.shape[0], 2)
        print_gpu_memory_stats("After retrieving features")
        return torch.from_numpy(features).float(), coords
