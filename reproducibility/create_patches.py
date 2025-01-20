# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage, get_best_level_for_downsample
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
from utils.file_utils import save_hdf5_coords
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
import cv2

# ###################################################################
# import bioformats
# import javabridge
# from utils.silence_javabridge_util import silence_javabridge
# ###################################################################
# print("Starting Java-Bridge...")
# javabridge.start_vm(class_path=bioformats.JARS, max_heap_size="2G")
# #silence javabridge debug messages
# jb_logger = silence_javabridge()
# ###################################################################

def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
					target_mpp=1.0,
				  patch_level = 0, # Always "patch" at level 0 but use microns per pixel (mpp) for influencing patch size
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None,
					storage_format='h5',
					return_all_patches=False):
	


	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide)) and slide.endswith('.czi')]
	print("warning: currently hardcoded to only search for .czi files.")
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
	else:
		print('Loading process list from {}'.format(process_list))
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

		# Check if there are any slides to process
		if df['process'].sum() == 0:
				raise ValueError("No slides are marked for processing in the 'process' column. Please set 'process' to 1 for at least one slide.")

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)
	print('total slides to process: {}'.format(total))

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in range(total): # for all WSIs
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i+1, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.' + storage_format)):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		# print("Initailsing WSI...")
		full_path = os.path.join(source, slide)
		WSI_object = WholeSlideImage(full_path)

		if return_all_patches:
			print(f"Generating all patches for {slide} without segmentation.")

			# Generate all patch coordinates based on your criteria
			all_patch_coords = WSI_object.generate_all_patch_coords(target_mpp=target_mpp,
																	patch_level=0,
																	patch_size=patch_size,
																	step_size=step_size)
			
			print(f"Number of patches: {len(all_patch_coords)}")
			# # print the first 10 elements
			# print(all_patch_coords[:10])
			# # print the 400th element and the following 10 elements
			# print(all_patch_coords[400:410])
			save_path_hdf5 = os.path.join(patch_save_dir, slide_id + '.' + storage_format)
			# Convert all_patch_coords to a NumPy array
			all_patch_coords_array = np.array(all_patch_coords, dtype=np.int64)

			#attr_dict = {'downsample': [1.0, 1.0], 
			#					'downsampled_level_dim': WSI_object.level_dim[0], 
			#					'name': slide_id, 
			#					'patch_level': 0, 
			#					'patch_size': patch_size, 
			#					'save_path': save_dir}
		
			attr = {'patch_size' :            patch_size, # To be considered...
							'patch_level' :           patch_level,
							'downsample':             WSI_object.level_downsamples[patch_level],
							'downsampled_level_dim' : tuple(np.array(WSI_object.level_dim[patch_level])),
							'level_dim':              WSI_object.level_dim[patch_level],
							'name':                   WSI_object.name,
							'save_path':              save_dir}
			
			attr_dict = { 'coords' : attr}

			print(f"Saving all patch coordinates to {save_path_hdf5}")
			save_hdf5_coords(output_path=save_path_hdf5, asset_dict={'coords': all_patch_coords_array}, attr_dict=attr_dict , mode='w', backend='zarr')
			
			continue  # Skip to the next slide
		


		# print("WSI initialised.")

		print("Value of use_default_params: ", use_default_params)


		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:    
				best_level = get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				best_level = get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		print("current_seg_params: ", current_seg_params)

		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			
			# Ensure all black pixels are turned white
			mask[np.all(mask == [0, 0, 0], axis=-1)] = [255, 255, 255]
			
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			cv2.imwrite(mask_path, mask)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'target_mpp': target_mpp , 'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
											'save_path': patch_save_dir, 'storage_format': storage_format})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.'+storage_format)
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
# REMOVE PATCH_LEVEL SINCE WE ALWAYS PATCH AT LEVEL 0 NOW
# parser.add_argument('--patch_level', type=int, default=0, 
					# help='downsample level at which to patch')
parser.add_argument('--target_mpp', type=float, default=0.25,
										help='microns per pixel')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')
parser.add_argument('--storage_format', type=str, default='h5', choices=['h5', 'zarr'],
                    help='format for storing the output files, choices are h5py (h5) or zarr')
parser.add_argument('--return_all_patches', action='store_true',
                    help='Bypass segmentation and return all patches')



if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	#patch_size and step_size must be resized to account for the target mpp




	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, step_size=args.step_size, 
											seg = args.seg,  use_default_params=False, save_mask = True, 
											stitch= args.stitch,
											target_mpp=args.target_mpp,
											# patch_level=args.patch_level,
											patch_level=0,
											patch = args.patch,
											process_list = process_list, auto_skip=args.no_auto_skip,
											storage_format=args.storage_format,
											return_all_patches=args.return_all_patches)
	
	print("End of program reached.")
	# print("Killing JavaBridge.")
	# javabridge.kill_vm()
