import pickle
import h5py
import zarr
import os
import numpy as np

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file

def save_hdf5_coords(output_path, asset_dict, attr_dict=None, mode='a', backend='h5', sample_count=0):
    # print(f"inside save_hdf5, output_path: {output_path}")
    try:
        #Print details about asset_dict and attr_dict
        # print(f"asset_dict: {asset_dict}")
        # print(f"attr_dict: {attr_dict}")
        if attr_dict is None:
            print(f"attr_dict is None?!?!?!?!!")

        if backend == 'h5':
            file = h5py.File(output_path, mode)
            for key, val in asset_dict.items():
                data_shape = val.shape
                if key not in file:
                    data_type = val.dtype
                    chunk_shape = (1, ) + data_shape[1:]
                    maxshape = (None, ) + data_shape[1:]
                    dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                    dset[:] = val
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                else:
                    dset = file[key]
                    dset.resize(len(dset) + data_shape[0], axis=0)
                    dset[-data_shape[0]:] = val
            file.close()
        elif backend == 'zarr':
            print(f"Saving to {output_path}")

            # Initialize the Zarr store
            store = zarr.DirectoryStore(output_path)
            root = zarr.open_group(store=store, mode=mode)

            for key, val in asset_dict.items():
                data_shape = val.shape
                data_type = val.dtype

                if key in root:
                    # Load the existing dataset
                    dset = root[key]

                    # Calculate the total shape to accommodate all new data at once
                    total_shape = (dset.shape[0] + data_shape[0],) + dset.shape[1:]

                    # Resize the dataset
                    dset.resize(total_shape)

                    # Write all the data at once
                    dset[-data_shape[0]:] = val
                else:
                    # Create a new dataset to accommodate all data
                    chunk_shape = data_shape  # Use the whole data as one chunk
                    dset = root.create_dataset(key, shape=data_shape, chunks=chunk_shape, dtype=data_type)

                    # Write the data
                    dset[:] = val

                    # Set attributes if provided
                    if attr_dict is not None and key in attr_dict:
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
                    

            # After writing all data, consolidate metadata for performance
            zarr.consolidate_metadata(store)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    except Exception as e:
        print(f"Error: {e}")
        raise e

    return output_path


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a', backend='h5', sample_count=0, batch_size=1, batch_id=0):
    try:
        if backend == 'h5':
            file = h5py.File(output_path, mode)
            for key, val in asset_dict.items():
                data_shape = (sample_count,) + val.shape[1:]
                data_type = val.dtype

                if key not in file:
                    chunk_shape = (1, ) + val.shape[1:]
                    dset = file.create_dataset(key, shape=data_shape, maxshape=data_shape, chunks=chunk_shape, dtype=data_type)
                else:
                    dset = file[key]

                # Calculate the start index for this batch
                start_index = batch_id * batch_size
                dset[start_index:start_index + val.shape[0]] = val

                if attr_dict is not None and key in attr_dict:
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
            file.close()
        elif backend == 'zarr':
            # print(f"Saving to {output_path}")

            store = zarr.DirectoryStore(output_path)
            root = zarr.open_group(store=store, mode=mode)

            for key, val in asset_dict.items():
                if key in root: # i.e. if the dataset already exists
                    dset = root[key]
                else: # i.e. if the dataset does not exist
                    data_shape = (sample_count,) + val.shape[1:]
                    data_type = val.dtype
                    chunk_shape = (batch_size,) + val.shape[1:]
                    dset = root.create_dataset(key, shape=data_shape, chunks=chunk_shape, dtype=data_type)

                # Calculate the start index for this batch
                start_index = batch_id * batch_size

                # Ensure we don't exceed the dataset size
                if start_index + val.shape[0] > sample_count:
                    raise ValueError("Trying to append more data than expected, start_index = {}, val.shape[0] = {}, sample_count = {}\n\n More info: key={}, shape={}, chunks={}, dtype={}".format(start_index, val.shape[0], sample_count, key, data_shape, chunk_shape, data_type))
                

                dset[start_index:start_index + val.shape[0]] = val
    except Exception as e:
        print(f"Error: {e}")
        raise e

    return output_path
