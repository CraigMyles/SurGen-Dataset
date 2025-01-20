# SurGen Dataset Reproducibility Guide

This guide outlines the steps necessary to reproduce the experiments and results described in the SurGen dataset repository/paper. The tissue segmentation and patch extraction pipeline is based on [CLAM](https://github.com/mahmoodlab/CLAM/)'s implementation which originally utilised [OpenSlide](https://openslide.org/api/python/). Significant changes have been made to enable compatibility with .CZI whole slide images (WSIs), incorporate pretrained patch-based feature extraction models, and transition from h5py to zarr. 

## Clone the Repository

```bash
git clone https://github.com/CraigMyles/SurGen-Dataset.git
```

## Create Python Environment

### Build Docker Image

Navigate to the `reproducibility/environment` directory and build the Docker image:

```bash
cd ./SurGen-Dataset/reproducibility/environment
make docker_image
```

### Run Docker Container

#### Adjust the Mounted Volume
Before running the container, ensure the location of the mounted volume is correctly configured in the Makefile. Then, execute:

```bash
make docker_run
```

#### Alternatively, Run Docker Manually

```bash
docker run --gpus all --shm-size 8G \
            --name $(LOCAL_USER)_$(DOCKER_IMAGE_NAME) \
            --user $(LOCAL_UID):$(LOCAL_GID) \
            -v /mnt/isilon1/craig:/home/$(LOCAL_USER)/data \
            -it $(DOCKER_IMAGE_NAME):v1
```

### Activate Conda Environment

If not already inside the environment:

```bash
source ~/miniconda3/bin/activate surgen
```
Note: Ensure Python version 3.8.18 is installed by running: ``python --version``.

### Navigate to the Repository

```bash
cd ~/SurGen-Dataset/reproducibility/
```

## Create Patches

This step identifies tissue areas within whole slide images (WSIs) for subsequent extraction. It uses tissue segmentation methods to remove background and isolate regions of interest.

### General Command

```bash
python create_patches.py \
--source ~/data/SurGen_WSIs \
--save_dir ~/data/SurGen_WSIs/zarr/one_point_zero_mpp/224 \
--patch_size 224 --step_size 224 --seg --patch --target_mpp 1.0 --storage_format zarr
```

### Using Tuned Parameters

For SurGen-specific segmentation parameters, you can utilise the tuned CSV files provided in the repository. Depending on the objective, you may only be utilising one of the two cohorts (SR386 or SR1482), otherwise you can run the ``Full SurGen Cohort`` method. 

#### SR1482 Cohort:

```bash
python create_patches.py --source ~/data/SR1482_WSIs/ \
--save_dir ~/data/SR1482_WSIs/zarr/one_point_zero_mpp/224 \
--process_list ~/SurGen-Dataset/reproducibility/dataset_csv/segment_params_SR1482.csv \
--patch_size 224 --step_size 224 --seg --patch --target_mpp 1.0 --storage_format zarr
```

#### SR382 Cohort:

```bash
python create_patches.py --source ~/data/SR382_WSIs/ \
--save_dir ~/data/SR382_WSIs/zarr/one_point_zero_mpp/224 \
--process_list ~/SurGen-Dataset/reproducibility/dataset_csv/segment_params_SR382.csv \
--patch_size 224 --step_size 224 --seg --patch --target_mpp 1.0 --storage_format zarr
```

#### Full SurGen Cohort:

```bash
python create_patches.py --source ~/data/SurGen_WSIs/ \
--save_dir ~/data/SurGen_WSIs/zarr/one_point_zero_mpp/224 \
--process_list ~/SurGen-Dataset/reproducibility/dataset_csv/segment_params_SurGen.csv \
--patch_size 224 --step_size 224 --seg --patch --target_mpp 1.0 --storage_format zarr
```

## Extract Features

This part uses the masks generated from the `create_patches.py` script to iterate across the whole slide image and extract features from non-overlapping tissue patches. These features can then be used for downstream machine learning tasks.

```bash
CUDA_VISIBLE_DEVICES=0 python extract_features.py \
--data_h5_dir ~/data/SurGen_WSIs/zarr/one_point_zero_mpp/224 \
--data_slide_dir ~/data/SurGen_WSIs \
--csv_path ~/SurGen-Dataset/reproducibility/dataset_csv/segment_params_SurGen.csv \
--feat_dir ~/data/SurGen_WSIs/zarr/one_point_zero_mpp/224 \
--batch_size 512 --target_patch_size=224 \
--slide_ext .czi --storage_format zarr --csv_shuffle \
--feature_extractor uni
```

## Train Weakly Supervised Model

Train a transformer-based model on WSI features for a specified task, such as predicting mismatch repair (MMR) or microsatellite instability (MSI) status.

Navigate to the ``./dataset_csv`` directory to view all predefined tasks. This directory includes training, validation, and test splits for tasks such as BRAF, KRAS, NRAS, and MSI, covering both the SR386 and SR1482 cohorts, as well as the combined SurGen dataset

If new tasks are required, they can be defined in the [./datasets/dataset_constants.py](./datasets/dataset_constants.py) file. Use the existing constructs in this file as a template for defining additional tasks.

```bash
CUDA_VISIBLE_DEVICES=0 python transformer_main_multiclass.py \
--train_fv_path=~/data/SurGen/train/zarr/one_point_zero_mpp/224/h5_files \
--val_fv_path=~/data/SurGen/validate/zarr/one_point_zero_mpp/224/h5_files \
--train_dataset_csv_path=~/SurGen-Dataset/reproducibility/dataset_csv/SurGen_msi_train.csv \
--val_dataset_csv_path=~/SurGen-Dataset/reproducibility/dataset_csv/SurGen_msi_validate.csv \
--task=MMR_MSI --cohort=SurGen --results_dir=~/data/vit_models/ \
--log_dir=./log_dir --epochs=5 --activation=relu --dropout=0.15 \
--feature_extractor=uni --lr=0.0001 --encoder_layers=2 --heads=2 --use_amp
```

## Evaluate the Model

Evaluate the trained model on the test dataset to obtain performance metrics such as AUROC.

The trained model weights from our paper can be downloaded from [Hugging Face](https://huggingface.co/craigmyles/surgen_msi_classification/tree/main). Ensure the ``--model_path`` argument points to the correct location.

```bash
CUDA_VISIBLE_DEVICES=0 python transformer_eval_script.py \
--test_fv_path=~/data/SurGen/test/zarr/one_point_zero_mpp/224/h5_files \
--test_dataset_csv_path=~/SurGen-Dataset/reproducibility/dataset_csv/SurGen_msi_test.csv \
--task=MMR_MSI --cohort=SurGen \
--model_path=~/SurGen-Dataset/reproducibility/models/best_msi_model_huggingface.pth \
--activation=relu \
--dropout=0.15 \
--feature_extractor=uni \
--encoder_layers=6 \
--nhead=2 \
--encoder_layers=2 \
--threshold=0.5
```

