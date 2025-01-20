import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
from models.owkin.rl_benchmarks.models import iBOTViT
from models.ctranspath.swin_transformer import swin_tiny_patch4_window7_224, ConvStem
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import zarr
from pylibCZIrw import czi as pyczi
from tqdm import tqdm
import hashlib
import gdown
from pathlib import Path
import wandb
import timm
from huggingface_hub import login, hf_hub_download
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Running on {device}")


def compute_w_loader(file_path, output_path, wsi, models,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, storage_format='h5'):
	start_time_overall = time.time()

	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""

	# if model has a transform attribute, assign that to custom_transforms in Whole_Slide_Bag_FP
	#! Turns out that all three models use the same transform, so this is not necessary
	# if hasattr(model, 'transform'):
	# 	model_transforms = model.transform
	# else:
	# 	model_transforms = None

	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_transforms=None,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size, storage_format=storage_format)
	x, y = dataset[0]
	kwargs = {'num_workers': 16, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	print(f"Dataset size: {len(dataset)}\n Batch size: {batch_size}\n Number of batches: {len(loader)}\n Samples per batch: {batch_size*len(loader)}\n Total samples: {len(dataset)}")


	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'

	# Create output paths if they don't exist already for each model consisting of output_path / model_name
	for name, model in models.items():
		model_output_path = os.path.join(os.path.dirname(output_path), f"{name}_{os.path.basename(output_path)}")
		os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
		# print(f"Created output path: {model_output_path}") # additional debug

	start_time_batch_collection = time.time()
	for count, (batch, coords) in enumerate(tqdm(loader)):
		# print(f"Batch {count} of {len(loader)}")
		end_time_batch_collection = time.time()
		# print(f"Batch collection time: {end_time_batch_collection - start_time_batch_collection} seconds; for count, (batch, coords) in enumerate(tqdm(loader))")
		start_time_batch_processing = time.time()
		with torch.no_grad():	
			start_time_to_device = time.time()
			batch = batch.to(device, non_blocking=True)
			# print("Batch on GPU: ", batch.is_cuda)
			end_time_to_device = time.time()
			# print(f"Time to device: {end_time_to_device - start_time_to_device} seconds")

			#time
			start_time_inference = time.time()
			# Extract features from the batch with each model
			# features = {}
			for name, model in models.items():
				features = model(batch).cpu().numpy()
				asset_dict = {'features': features, 'coords': coords}
				#output patch must include the model name
				model_output_path = os.path.join(os.path.dirname(output_path), name, os.path.basename(output_path)) # ./h5_files/model_name/wsi_name.zarr
				# print(f"DEBUG: ABOUT TO RUN SAVE_HDF5 WITH MODEL_OUTPUT_PATH: {model_output_path}\n ASSET_DICT: {asset_dict}\n ATTR_DICT: None\n MODE: {mode}\n BACKEND: {storage_format}\n SAMPLE_COUNT: {len(dataset)}")
				save_hdf5(model_output_path, asset_dict, attr_dict= None, mode='a', backend=storage_format, sample_count=len(dataset), batch_size=batch_size, batch_id=count)

			# features = model(batch)
			end_time_inference = time.time()
			# print(f"Inference time: {end_time_inference - start_time_inference} seconds")
			start_time_batch_collection = time.time()
		

	end_time_overall = time.time()
	print(f"Total time taken: {end_time_overall - start_time_overall} seconds (end of compute_w_loader))")

	#wandb logging
	wandb.log({
		"WSI ID": os.path.basename(output_path),
		"Processing time": end_time_overall - start_time_overall,
		"Batch Size": batch_size,
		"Number of Batches": len(loader),
		"Total Samples": len(dataset),
	})

	output_path = model_output_path
	return output_path

def download_and_validate_weights(weights_path: Path, download_url: str, expected_hash: str = None) -> None:
    """
    Downloads and optionally validates the weights using their SHA-256 hash.
    
    Parameters:
        weights_path (Path): Path to store downloaded weights.
        download_url (str): URL to download weights from.
        expected_hash (str, optional): Expected SHA-256 hash of weights.
        
    Raises:
        ValueError: If the downloaded file's hash doesn't match the expected hash.
    """
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(download_url, str(weights_path), quiet=False)

    if expected_hash:
        sha256 = hashlib.sha256()
        with weights_path.open("rb") as f:
            while True:
                data = f.read(1 << 16)
                if not data:
                    break
                sha256.update(data)

        if sha256.hexdigest() != expected_hash:
            raise ValueError("Invalid weights file")

def load_models(feature_extractors):
	"""
	Function to load the model based on the feature extractor specified.
	"""
	models = {}
	for feature_extractor in feature_extractors:
		if feature_extractor == 'resnet50':
			#!Load ResNet50
			print("ResNet50 model successfully initialised...")
			models['resnet50'] = resnet50_baseline(pretrained=True)
			# return resnet50_baseline(pretrained=pretrained)
		elif feature_extractor == 'resnet50-b':
			# raise a NotImplementedError with guidance for implementation
			raise NotImplementedError("RESNET50-B is not currently implemented. If you wish to implement it, "
														 		"consider modifying the standard ResNet-50 by removing the 4th layer. "
														 		"This adjustment could potentially align with the specific characteristics "
														 		"of RESNET50-B you're aiming to achieve.")

		elif feature_extractor == 'owkin':
			#!Load Owkin
			# weights_path = Path("/home/cggm1/data/pretrained_models/owkin.pth") # apologies about the hardcoding
			weights_path = Path.home() / "data" / "pretrained_models" / "owkin.pth"

			if not weights_path.exists():
				download_and_validate_weights(
					weights_path,
					"https://drive.google.com/u/0/uc?id=1uxsoNVhQFoIDxb4RYIiOtk044s6TTQXY&export=download",
					"3bc6e4e353ebdd75b31979ff470ffa4d67349828057957dcc8d0f13e9d224d3f"
				)
			models['owkin'] = iBOTViT(architecture="vit_base_pancan", encoder="student", weights_path=str(weights_path))
			print("Owkin model successfully initialised...")
		elif feature_extractor == 'ctranspath':
			#!Load CTransPath
			# weights_path = Path("/home/cggm1/data/pretrained_models/ctranspath.pth") # Change this path to the directory where you want to save the weights
			weights_path = Path.home() / "data" / "pretrained_models" / "ctranspath.pth"

			if not weights_path.exists():
				download_and_validate_weights(
					weights_path,
					"https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX&export=download",
					"7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539"
				)
			
			# from https://github.com/Avic3nna/STAMP/blob/57872a0cc9289933f4943e4025bbeefe88ca551e/stamp/preprocessing/helpers/feature_extractors.py#L37-L46C2
			model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
			model.head = nn.Identity()

			ctranspath = torch.load(weights_path, map_location=torch.device('cpu'))
			model.load_state_dict(ctranspath['model'], strict=True)

			models['ctranspath'] = model
			print("CTransPath model successfully initialised...")
		elif feature_extractor == 'uni':
			#!Load MahmoodLab/UNI model
			# Define the directory where you want to save the model weights
			local_dir = Path.home() / "data" / "pretrained_models" # Change this path to the directory where you want to save the weights
			# Ensure the directory exists
			local_dir.mkdir(parents=True, exist_ok=True)
			# Specify the path where you want to save the weights file
			weights_path = local_dir / "uni.pth"

			# Check if the model weights already exist
			if not weights_path.exists():
					print(f"Downloading weights for {feature_extractor} model...")
					# Download the file and save it as "uni.pth"
					#login to huggingface
					login()
					hf_hub_download(
							repo_id="MahmoodLab/UNI",
							filename="pytorch_model.bin",
							cache_dir=local_dir,  # Specify where to save the file
							force_filename="uni.pth"  # Force the downloaded file to be saved as "uni.pth"
					)
					print(f"Downloaded {feature_extractor} model weights to {weights_path}")
			else:
					print(f"Weights for {feature_extractor} model found at {weights_path}")
			
			# Load the model weights
			model = timm.create_model(
				"vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
			)
			model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
			models['uni'] = model

		else:
			raise ValueError('Invalid feature_extractor name')
	return models

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--csv_shuffle', default=False, action='store_true')
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--storage_format', type=str, default='h5', choices=['h5', 'zarr'],
                    help='format for storing the output files, choices are h5py (h5) or zarr')
parser.add_argument('--feature_extractor', nargs='+', default=['resnet50'], 
                    choices=['resnet50', 'resnet50-b', 'owkin', 'ctranspath', 'uni'],
                    help='List of feature extractors to use')

args = parser.parse_args()


if __name__ == '__main__':

	#spinning up wandb to log gpu usage
	wandb.init(project="feature-extraction", tags=["feature-extraction"], config={
		"data_h5_dir": args.data_h5_dir,
		"data_slide_dir": args.data_slide_dir,
		"slide_ext": args.slide_ext,
		"csv_path": args.csv_path,
		"csv_shuffle": args.csv_shuffle,
		"feat_dir": args.feat_dir,
		"batch_size": args.batch_size,
		"no_auto_skip": args.no_auto_skip,
		"custom_downsample": args.custom_downsample,
		"target_patch_size": args.target_patch_size,
		"storage_format": args.storage_format,
		"feature_extractor": args.feature_extractor
	})

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError
	
	if args.csv_shuffle:
		print('CSV_SHUFFLE ENABLED: shuffling csv')

	bags_dataset = Dataset_All_Bags(csv_path, shuffle=args.csv_shuffle)

	storage_format = args.storage_format
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint(s)')
	# model = resnet50_baseline(pretrained=True)
	models = load_models(args.feature_extractor)
	for name, model in models.items():
			model = model.to(device)
			if torch.cuda.device_count() > 1:
					models[name] = nn.DataParallel(model)
			models[name].eval()
			print(f"Loaded model {name} on GPU: {next(model.parameters()).is_cuda}")

	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.'+storage_format
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx+1, total))
		print(slide_id)

		#skip if already exists
		if not args.no_auto_skip and slide_id+'.pt' in dest_files: 
			print('skipped {}'.format(slide_id))
			continue 

		#skip if in progress (check for .zarr or .h5 for all models)
		for name, model in models.items():
			model_name_plus_output_path = os.path.join(os.path.dirname(args.feat_dir), name, os.path.basename(args.feat_dir), bag_name)
			# print(f"debug: checking if {model_name_plus_output_path} exists")
			if os.path.exists(model_name_plus_output_path):
				print('skipped {}\n Info: {} exists. (BUT THE .PT DIDNT EXIST.)'.format(slide_id, model_name_plus_output_path))
				continue


		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		# wsi = openslide.open_slide(slide_file_path)
		# wsi = bioformats.ImageReader(slide_file_path)
		wsi = pyczi.CziReader(slide_file_path)

		output_file_path = compute_w_loader(h5_file_path, 
																			output_path, 
																			wsi, 
																			models = models, 
																			batch_size = args.batch_size, 
																			verbose = 1, 
																			print_every = 20, 
																			custom_downsample=args.custom_downsample, 
																			target_patch_size=args.target_patch_size,
																			storage_format=args.storage_format
																			)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		
		if storage_format == 'h5':
			file = h5py.File(output_file_path, "r")
		elif storage_format == 'zarr':
			print (f"Using Zarr to load _file_ from: {output_file_path}")
			file = zarr.open_group(output_file_path, mode='r')
		else:
			raise ValueError('Invalid storage_format specified.')



		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)

		print(f"About to try to save features to {os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt')}")
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
