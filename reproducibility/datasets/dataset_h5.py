from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import cv2
import math
import re
import pdb
import pickle

from pathlib import Path
from typing import Union, Optional

# import bioformats
# import javabridge

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py
import zarr
import time

from random import randrange

def series(int):
    return 1/(2**int)

def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)

    trnsfrms_val = transforms.Compose(
                    [
                     transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
        file_path,
        wsi,
        pretrained=False,
        custom_transforms=None,
        custom_downsample=1,
        target_patch_size=-1,
        storage_format='h5',
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained=pretrained
        self.wsi = wsi
        self.offset = self.wsi.total_bounding_rectangle[0], self.wsi.total_bounding_rectangle[1]
        # print(self.wsi.) 
        self.storage_format = storage_format

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
            print("LOG: Using eval_transforms")
        else:
            self.roi_transforms = custom_transforms
            print("LOG: Using custom_transforms")

        self.file_path = file_path

        # Initialize relevant context manager based on storage format
        if self.storage_format == 'h5':
            # print("inside if self.storage_format == 'h5':")
            context_manager = h5py.File(self.file_path, "r")
        elif self.storage_format == 'zarr':
            # print("inside elif self.storage_format == 'zarr':")
            context_manager = zarr.open(self.file_path, mode='r')
        else:
            raise ValueError("Invalid storage_format: choose either 'h5' or 'zarr'")

        # Use the chosen context manager
        timer_start_use_context_manager = time.time()
        with context_manager as f:
                dset = f['coords']
                self.patch_level = f['coords'].attrs['patch_level']
                self.patch_size = f['coords'].attrs['patch_size']
                self.length = len(dset)
                if target_patch_size > 0:
                        self.target_patch_size = (target_patch_size, ) * 2 # creates target_patch_size=(target_patch_size, target_patch_size)
                elif custom_downsample > 1:
                        self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
                else:
                        self.target_patch_size = None
        timer_end_use_context_manager = time.time()
        print("LOG: Time to use context manager: {}".format(timer_end_use_context_manager - timer_start_use_context_manager))
        self.summary()

            
    def __len__(self):
        return self.length

    def summary(self):
        if self.storage_format == 'h5':
                context_manager = h5py.File(self.file_path, "r")
        elif self.storage_format == 'zarr':
                context_manager = zarr.open_group(self.file_path, mode='r')
        else:
                raise ValueError("Invalid storage_format: choose either 'h5' or 'zarr'")
                
        with context_manager as f:
                dset = f['coords']
                for name, value in dset.attrs.items():
                        print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        # timer_start_getitem = time.time()

        # timer_start_get_coords = time.time()
        if self.storage_format == 'h5':
            with h5py.File(self.file_path, 'r') as hdf5_file:
                coord = hdf5_file['coords'][idx]
        elif self.storage_format == 'zarr':
            with zarr.open(self.file_path, mode='r') as zarr_file:
                coord = zarr_file['coords'][idx]
        else:
            raise ValueError("Invalid storage_format: choose either 'h5' or 'zarr'")
        # timer_end_get_coords = time.time()
        # print("LOG: Time to get coords: {}".format(timer_end_get_coords - timer_start_get_coords))

        # _XYWH = (coord[0]/64,coord[1]/64,self.patch_size,self.patch_size)
        # img = np.array(self.wsi.read(series=self.patch_level, XYWH=_XYWH))
        
        # timer_start_get_img = time.time()
        _XYWH =(self.offset[0]+coord[0],
                        self.offset[1]+coord[1],
                        int(self.patch_size/series(self.patch_level)),
                        int(self.patch_size/series(self.patch_level))) #xywh

        # print("roi={}, zoom={}".format(_XYWH, series(self.patch_level)))
        # img = np.array(self.wsi.read(roi=_XYWH, zoom=series(self.patch_level))) # We want to always use zoom 1.0 for the WSI, so we need to downsample the patch_size
        img = np.array(self.wsi.read(roi=_XYWH, zoom=1.0))


        if self.target_patch_size is not None:
            # print("inside if self.target_patch_size is not None:")
            # img = img.resize(self.target_patch_size) # This uses numpy which is not ideal for images.
            # resize image to target_patch_size using cv2 downsize interpolation
            # print("LOG: img.shape: {}, self.target_patch_size: {}".format(img.shape, self.target_patch_size))
            img = cv2.resize(img, self.target_patch_size, interpolation=cv2.INTER_AREA) # THIS MIGHT BE BETTER DONE ON THE GPU AS PART OF THE TRANSFORMS
        # print("about to do the roi_transforms")
        img = self.roi_transforms(img).unsqueeze(0)
        # timer_end_get_img = time.time()
        # print("LOG: Time to get img: {}".format(timer_end_get_img - timer_start_get_img))

        # timer_end_getitem = time.time()
        # print("LOG: Time to getitem: {}".format(timer_end_getitem - timer_start_getitem))
        return img, coord

class Dataset_All_Bags(Dataset):
    def __init__(self, csv_path, shuffle=False):
        self.df = pd.read_csv(csv_path)
        if shuffle:
            # Shuffles the DataFrame in place
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]
        

# Class to handle zarr data (sets of WSI feature vectors. Each WSI has its own zarr file). 
class SlideDataset(Dataset):
    def __init__(
        self,
        slide: Union[str, Path],
        transform,
        inverse_transform,
        batch_size: Optional[int] = None,
    ):

        slide = Path(slide)
        #storage format determined by file extension found in wsi_vector_path
        self.storage_format = slide.suffix
        if self.storage_format not in ['zarr']:
            raise ValueError("Invalid storage_format: implemented only for zarr at present.")
        self.zarr_group = zarr.open_group(str(slide), mode="r")
        self.batch_size = batch_size
        self.num_patches = self.zarr_group["patches"].shape[0]
        self.num_batches = math.ceil(self.num_patches / self.batch_size) if self.batch_size else 1

        self.name = slide.stem
        self.transform = transform
        self.inverse_transform = inverse_transform

    def __len__(self):
        return len(self.num_batches)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.num_patches)
        coords = self.zarr_group["coords"][start:end]
        features = self.zarr_group["features"][start:end]
        features = self.transform(features)

        return features, torch.from_numpy(coords)