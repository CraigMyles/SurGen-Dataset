import argparse
from datasets.dataset_constants import TASKS, AVAILABLE_COHORTS, LABEL_ENCODINGS
from pathlib import Path
from datasets.dataset_loader import Dataset_All_Bags, SlideDataset
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import torchmetrics
import numpy as np
import glob
import os
from typing import Callable, Tuple
from torch import Tensor
import logging

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_cached_memory_in_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 ** 3)
    return 0

def print_gpu_memory_stats(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        cached = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert bytes to GB
        logging.debug(f"{tag} - Allocated memory: {allocated:.4f} GB, Cached memory: {cached:.4f} GB")
    else:
        logging.debug(f"{tag} - CUDA is not available. No GPU detected.")

class TransformerForClassification(nn.Module):
    def __init__(self, fv_extractor_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="gelu", layer_norm_eps=1e-5, num_classes=4):
        super(TransformerForClassification, self).__init__()
        self.fc = nn.Linear(fv_extractor_size, d_model)
        self.relu = nn.ReLU()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation, layer_norm_eps=layer_norm_eps)
        #if num_classes is 2, binary classification, else multiclass classification
        self.num_classes = num_classes
        if self.num_classes == 2:
            self.classifier = nn.Linear(d_model, 1) # Binary classification, enables use of BCEWithLogitsLoss and Sigmoid activation
        elif self.num_classes > 2:
            self.classifier = nn.Linear(d_model, self.num_classes) # Multiclass classification, enables use of CrossEntropyLoss and Softmax activation
        else:
            raise ValueError(f"Invalid number of classes: {num_classes}. Must be greater than 1.")

    def forward(self, src):
        # print("Transformer Type:", type(self.transformer))
        # print("Encoder Callable:", callable(self.transformer.encoder))
        # src expected shape: (sequence_length, batch_size, feature vector dim)
        src = self.relu(self.fc(src))  # Apply the fully connected layer and ReLU
        # print("Memory shape before encoder:", src.shape)
        memory = self.transformer.encoder(src)  # Now src is of shape (sequence_length, batch_size, d_model)
        # print("Memory shape after encoder:", memory.shape)  # Debug: Should be [sequence_length, batch_size, d_model]
        out = self.classifier(memory.mean(dim=0))  # Mean pooling across the sequence_length dimension
        # print("Output shape:", out.shape)
        return out if self.num_classes > 2 else out.view(-1)
        # return out if self.num_classes > 2 else out.view(-1, 1)

def delete_previous_best_models(directory, metric_name):
    pattern = str(directory / f"best_model_{metric_name}_epoch_*")
    for filepath in glob.glob(pattern):
        try:
            os.remove(filepath)
            print(f"Deleted previous best model: {filepath}")
        except Exception as e:
            print(f"Error deleting file {filepath}: {e}")

def config_train_step(model: nn.Module, criterion: nn.Module, num_classes: int) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    print(f"Configuring train step for {num_classes} classes...")
    if num_classes == 2:
        print(f"Configuring train step for binary classification.")
        def train_step(features: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            # print(f"Features shape: {features.shape}, Features dtype: {features.dtype}, about to run through model.")
            logits = model(features)
            # print(f"Logits shape: {logits.shape}, Logits dtype: {logits.dtype}")
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            return loss, labels, probs, preds
    else:
        print(f"Configuring train step for multiclass classification.")
        def train_step(features: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            logits = model(features)
            logits = logits.unsqueeze(0)
            labels = labels.long()
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            return loss, labels, probs, preds
    return train_step

def main():
    parser = argparse.ArgumentParser(description='Train a classifier on feature vectors.')
    parser.add_argument('--train_fv_path', type=str, help='path to training feature vectors (zarr) !Please point to the h5_files directory!')
    parser.add_argument('--val_fv_path', type=str, help='path to val feature vectors (zarr) !Please point to the h5_files directory!')
    parser.add_argument('--train_dataset_csv_path', type=str, help='path to train dataset_csv (location full of csv tasks)')
    parser.add_argument('--val_dataset_csv_path', type=str, help='path to validation dataset_csv (location full of csv tasks)')
    parser.add_argument('--task', type=str, help='task to train classifier for')
    parser.add_argument('--cohort', type=str, help='cohort to train classifier for', choices=AVAILABLE_COHORTS)
    parser.add_argument('--results_dir', type=str, help='path to directory to save results', default="~/data/vit_models")
    parser.add_argument('--log_dir', type=str, help='path to directory to save logs')
    parser.add_argument('--epochs', type=int, help='number of epochs to train for', default=100)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and validation (default: 1)')
    parser.add_argument('--dropout', type=float, help='dropout rate', default=0.1)
    parser.add_argument('--activation', type=str, help='activation function', default="gelu", choices=["gelu", "relu"])
    parser.add_argument('--lr', type=float, help='learning rate', default=0.000001)
    parser.add_argument('--encoder_layers', type=int, help='number of encoder layers', default=6)
    parser.add_argument('--heads', type=int, help='number of transformer heads. Default 4 but torch default suggests 8.', default=4)
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--balance_classes', action='store_true', help='balance the classes in loss calculation')
    parser.add_argument('--feature_extractor', type=str, help='choose which feature extractor to use', default="ctranspath", choices=["ctranspath", "owkin", "resnet50", "resnet50-b", "uni"])
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging for debugging.')

    args = parser.parse_args()

    setup_logging(args.verbose)
    print_gpu_memory_stats("Beginning of script")

    # Validate the task for the selected cohort
    valid_tasks_for_cohort = TASKS.get(args.cohort, [])
    if args.task not in valid_tasks_for_cohort:
        valid_tasks_str = ', '.join(valid_tasks_for_cohort)
        parser.error(f"The task '{args.task}' is not valid for cohort '{args.cohort}'. "
                     f"Valid tasks for this cohort are: {valid_tasks_str}")

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set automatic mixed precision (AMP) if enabled
    use_amp = args.use_amp
    if use_amp:
        print("Automatic Mixed Precision (AMP) enabled")
        scaler = GradScaler()

    train_fv_path = Path(args.train_fv_path)
    val_fv_path = Path(args.val_fv_path)
    train_dataset_csv_path = Path(args.train_dataset_csv_path)
    val_dataset_csv_path = Path(args.val_dataset_csv_path)
    task = args.task
    cohort = args.cohort
    # Override results_dir if set to "./results"
    if args.results_dir == "./results":
        args.results_dir = "~/data/vit_models/"
    results_dir = Path(args.results_dir)
    log_dir = Path(args.log_dir)
    feature_extractor = args.feature_extractor

    # add the fv extractor name to the end of the tran and val paths
    train_fv_path = train_fv_path / feature_extractor
    val_fv_path = val_fv_path / feature_extractor

    # Validate the paths
    for path in [train_fv_path, val_fv_path, train_dataset_csv_path, val_dataset_csv_path]:
        if not path.exists():
            # parser.error(f"Path '{path}' does not exist")
            print(f"Path '{path}' does not exist, but ignoring it anyway")
    # Create the results and log dirs if they don't exist
    for path in [results_dir, log_dir]:
        if not path.exists():
            path.mkdir(parents=True)

    print("----------------------------------------")
    print(f"train_fv_path: {train_fv_path}")
    print(f"val_fv_path: {val_fv_path}")
    print(f"train_dataset_csv_path: {train_dataset_csv_path}")
    print(f"val_dataset_csv_path: {val_dataset_csv_path}")
    print(f"task: {task}")
    print(f"cohort: {cohort}")
    print(f"Label Keys: {LABEL_ENCODINGS[cohort][task]}")
    print(f"results_dir: {results_dir}")
    print(f"log_dir: {log_dir}")
    print("----------------------------------------")


    #if feature extractor is ctranspath or owkin, set ft_extractor_size to 768, if resnet50-b or uni, set to 1024. For ResNet-50, set to 2048.
    print(f"Feature extractor before assignment: {feature_extractor}")
    if feature_extractor == "ctranspath" or feature_extractor == "owkin":
        ft_extractor_size = 768
    elif feature_extractor == "resnet50-b" or feature_extractor == "uni":
        ft_extractor_size = 1024
    elif feature_extractor == "resnet50":
        ft_extractor_size = 2048
    print(f"Feature extractor size after assignment: {ft_extractor_size}")


    #init torch transformer model #! Parameterise these
    d_model = 512 #(default=512). The number of expected features in the encoder/decoder inputs.
    nhead = args.heads #(default=8). The number of heads in the multiheadattention models.
    num_encoder_layers = args.encoder_layers #(default=6). The number of sub-encoder-layers in the encoder.
    num_decoder_layers = 6 #(default=6). The number of sub-decoder-layers in the decoder.
    dim_feedforward = 2048 #(default=2048). The dimension of the feedforward network model.
    dropout = args.dropout #! Parameterised.
    activation = args.activation #! Parameterised.
    layer_norm_eps = 1e-5 #(default=1e-5).

    print(f"About to initialise TransformerForClassification model with the following parameters: \n")
    print(f"num_classes: {len(LABEL_ENCODINGS[cohort][task])}")

    num_classes = len(LABEL_ENCODINGS[cohort][task])  # Correct way to set the number of classes

    print(f"num_classes: {num_classes}")

    transformer_model = TransformerForClassification(
        fv_extractor_size=ft_extractor_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        num_classes=len(LABEL_ENCODINGS[cohort][task])
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model.to(device)

    # Initialize train_dataset_all_bags
    train_dataset_all_bags = Dataset_All_Bags(csv_path=train_dataset_csv_path, task=task, cohort=cohort, shuffle=False)
    unique_labels_train = train_dataset_all_bags.df['label'].unique()
    print(f"Unique train labels for criterion: {unique_labels_train}")

    # Initialize val_dataset_all_bags
    val_dataset_all_bags = Dataset_All_Bags(csv_path=val_dataset_csv_path, task=task, cohort=cohort, shuffle=False)
    unique_labels_val = val_dataset_all_bags.df['label'].unique()
    print(f"Unique val labels for criterion: {unique_labels_val}")

    #print counts for each label in train and val datasets
    # print(f"Train dataset label counts: \n{train_dataset_all_bags.df['label'].value_counts()}")
    # print(f"Val dataset label counts: \n{val_dataset_all_bags.df['label'].value_counts()}")
    # Convert label counts to a DataFrame for train dataset
    train_label_counts = train_dataset_all_bags.df['label'].value_counts().reset_index()
    train_label_counts.columns = ['Label', 'Count']
    total_train_samples = len(train_dataset_all_bags.df)
    train_label_counts['Percentage'] = (train_label_counts['Count'] / total_train_samples) * 100

    # Convert label counts to a DataFrame for val dataset
    val_label_counts = val_dataset_all_bags.df['label'].value_counts().reset_index()
    val_label_counts.columns = ['Label', 'Count']
    total_val_samples = len(val_dataset_all_bags.df)
    val_label_counts['Percentage'] = (val_label_counts['Count'] / total_val_samples) * 100

    # Display label counts and percentages using Pandas DataFrame
    print("Train dataset label counts and percentages:")
    print(train_label_counts)
    print("\nVal dataset label counts and percentages:")
    print(val_label_counts)

    #assert that unique labels in train and val datasets are the same, regardless of order
    assert set(unique_labels_train) == set(unique_labels_val), f"Unique labels in train and val datasets do not match: {unique_labels_train} vs {unique_labels_val}"

    wsi_count = len(train_dataset_all_bags)
    print(f"Total operable number of WSIs associated with {train_dataset_csv_path}: {wsi_count}.")
    print(f"Total operable number of WSIs associated with {val_dataset_csv_path}: {len(val_dataset_all_bags)}.")

    # Define the loss function and optimiser

    # Calculate the positive class weight for balancing
    # Assume binary labels with 1 for positive and 0 for negative
    num_positives = train_dataset_all_bags.df['label'].sum()
    num_negatives = len(train_dataset_all_bags) - num_positives
    pos_weight = torch.tensor([num_negatives / num_positives], device=device)

    train_label_counts['Weight'] = total_train_samples / (train_label_counts['Count'] * len(train_label_counts))
    class_weights_tensor = torch.tensor(train_label_counts['Weight'].values, dtype=torch.float, device=device)

    # print(f"class_weights tensor: {class_weights_tensor.cpu().detach().numpy()}")

    # Calculate class weights for balancing if needed
    if args.balance_classes:
        label_counts = train_dataset_all_bags.df['label'].value_counts()
        total_count = label_counts.sum()
        class_weights = total_count / label_counts
        class_weights_tensor = class_weights.sort_index().values  # Ensure ordering of weights matches label indices
        class_weights_tensor = torch.tensor(class_weights_tensor, dtype=torch.float, device=device)

    # Define the loss function with class balancing and type of classification
    if num_classes == 2:
        # Binary classification
        pos_weight = pos_weight if args.balance_classes else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        print(f"Using BCEWithLogitsLoss as the loss function. Pos weight: {pos_weight}")
    else:
        # Multiclass classification
        class_weights = class_weights_tensor if args.balance_classes else None
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        print(f"Using CrossEntropyLoss as the loss function. Class weights: {class_weights_tensor.cpu().detach().numpy()}")

    print(f"Using {'BCEWithLogitsLoss' if num_classes == 2 else 'CrossEntropyLoss'} as the loss function.")
    print(f"Class balancing is {'enabled' if args.balance_classes else 'disabled'}.")


    learning_rate = args.lr #! Parameterised!
    optimiser = torch.optim.Adam(transformer_model.parameters(), lr=learning_rate) #! Parameterise this
    
    # num_epochs = 1000 #! Parameterise this
    num_epochs = args.epochs #! Parameterised!
    # batch_size = 1# 2**13 #! Parameterise this
    batch_size = args.batch_size #! Parameterised!

    # Set the train/val step function based on the number of classes. i.e. sigmoid vs softmax and dimension handling
    process_batch = config_train_step(transformer_model, criterion, num_classes) 

    #weights and biases initialisation and logging
    wandb.init(project="vit-path", tags=[task, cohort, feature_extractor], config={
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "activation": activation,
        "layer_norm_eps": layer_norm_eps,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "criterion": criterion,
        "optimiser": optimiser,
        "task": task,
        "cohort": cohort,
        "balance_classes": args.balance_classes,
        "learning_rate": learning_rate,
        "seed": args.seed,
        "use_amp": use_amp
    })
    run_id = wandb.run.id

    # Define the directory structure for model outputs
    run_dir = results_dir / cohort / task / feature_extractor / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"Permission denied: unable to create directory {run_dir}. Error: {e}")
    except FileExistsError as e:
        print(f"The directory {run_dir} already exists. Error: {e}")
    except Exception as e:
        print(f"An error occurred while creating directory {run_dir}. Error: {e}")

    # Define paths for model saving based on the new directory structure
    model_save_path_loss = run_dir / 'best_model_val_loss_epoch_{epoch}_loss_{loss:.4f}.pth'
    model_save_path_auroc = run_dir / 'best_model_val_auroc_epoch_{epoch}_auroc_{auroc:.4f}.pth'


    #prrint unique labels train for sanity check
    print(f"Unique labels in train dataset: {unique_labels_train}")
    print(f"Len of unique labels in train dataset: {len(unique_labels_train)}")
    print(f"Type of unique labels in train dataset: {type(unique_labels_train)}")
    print(f"Type of len of unique labels in train dataset: {type(len(unique_labels_train))}")

    #initialise torchmetrics
    num_classes = len(unique_labels_train)
    torchmetric_task = "binary" if num_classes==2 else "multiclass"
    train_accuracy = torchmetrics.Accuracy(task=torchmetric_task, num_classes=num_classes).to(device)
    val_accuracy = torchmetrics.Accuracy(task=torchmetric_task, num_classes=num_classes).to(device)
    train_auroc = torchmetrics.AUROC(task=torchmetric_task, num_classes=num_classes).to(device)
    val_auroc = torchmetrics.AUROC(task=torchmetric_task, num_classes=num_classes).to(device)
    train_precision = torchmetrics.Precision(task=torchmetric_task, num_classes=num_classes).to(device)
    val_precision = torchmetrics.Precision(task=torchmetric_task, num_classes=num_classes).to(device)
    train_recall = torchmetrics.Recall(task=torchmetric_task, num_classes=num_classes).to(device)
    val_recall = torchmetrics.Recall(task=torchmetric_task, num_classes=num_classes).to(device)
    # train_confusion_matrix = torchmetrics.ConfusionMatrix(task=torchmetric_task, num_classes=num_classes).to(device)
    # val_confusion_matrix = torchmetrics.ConfusionMatrix(task=torchmetric_task, num_classes=num_classes).to(device)
    

    # Initialize a dictionary to track the best metrics and their corresponding epochs
    best_metrics = {
        "val_loss": {"value": float('inf'), "epoch": -1},
        "val_accuracy": {"value": 0.0, "epoch": -1},
        "val_precision": {"value": 0.0, "epoch": -1},
        "val_recall": {"value": 0.0, "epoch": -1},
        "val_auroc": {"value": 0.0, "epoch": -1},
    }

    print_gpu_memory_stats("Before starting training")

    # Run dummy batch to initialise the model, this is necessary to avoid a CUDA out of memory error.
    import zarr
    # Get largest number of patches in a slide (consider all slides) and print this value as well as which slide (zarr file) it belongs to
    largest_num_patches = 0
    largest_num_patches_slide_id = ""
    for slide_id, _ in train_dataset_all_bags:
        num_patches = zarr.open(str(train_fv_path / f"{slide_id}.zarr"))['features'].shape[0]
        if num_patches > largest_num_patches:
            largest_num_patches = num_patches
            largest_num_patches_slide_id = slide_id
    print(f"Largest number of patches in a slide: {largest_num_patches} for slide: {largest_num_patches_slide_id}")

    #do the same for the val dataset
    largest_num_patches_val = 0
    largest_num_patches_slide_id_val = ""
    for slide_id, _ in val_dataset_all_bags:
        num_patches = zarr.open(str(val_fv_path / f"{slide_id}.zarr"))['features'].shape[0]
        if num_patches > largest_num_patches_val:
            largest_num_patches_val = num_patches
            largest_num_patches_slide_id_val = slide_id
    print(f"Largest number of patches in a slide (val dataset): {largest_num_patches_val} for slide: {largest_num_patches_slide_id_val}")

    # Run dummy batch to initialise the model, this is necessary to avoid a CUDA out of memory error.
    largest_train_num_patches = max([zarr.open(str(train_fv_path / f"{slide_id}.zarr"))['features'].shape[0] for slide_id, _ in train_dataset_all_bags])
    largest_val_num_patches = max([zarr.open(str(val_fv_path / f"{slide_id}.zarr"))['features'].shape[0] for slide_id, _ in val_dataset_all_bags])
    largest_num_patches = max(largest_train_num_patches, largest_val_num_patches)
    dummy_features = torch.randn(23000, ft_extractor_size, dtype=torch.float32).to(device)
    dummy_labels = torch.zeros(1).to(device)

    #print info on features and lables, they will need to be moved away from gpu before..
    logging.debug(f"Dummy Features shape: {dummy_features.shape}, Dummy Features dtype: {dummy_features.dtype}")
    logging.debug(f"Dummy Labels shape: {dummy_labels.shape}, Dummy Labels dtype: {dummy_labels.dtype}")
    #print small sample of features and labels
    logging.debug(f"Sample of dummy features: {dummy_features[:5]}")
    # print(f"Sample of dummy labels: {dummy_labels[:5]}")

    logging.debug("Running dummy batch to allocate memory...")

    logging.debug(f"Going to try to allocate a total of {largest_num_patches  * ft_extractor_size * 4 / (1024 ** 3):.2f} GB of memory.")

    with autocast(enabled=use_amp):
        print_gpu_memory_stats("Before dummy batch")
        dummy_loss, _, _, _ = process_batch(dummy_features, dummy_labels)
        print_gpu_memory_stats("after process_batch")
    if use_amp:
        logging.debug("Running dummy batch with AMP")
        scaler.scale(dummy_loss).backward()
        scaler.step(optimiser)
        scaler.update()
        print_gpu_memory_stats("after dummy batch with AMP")
    else:
        dummy_loss.backward()
        optimiser.step()
    del dummy_features, dummy_labels, dummy_loss
    torch.cuda.empty_cache()
    logging.debug("Memory allocated.")

    # Training loop
    for epoch in range(num_epochs):  # num_epochs is the number of epochs
        print_gpu_memory_stats(f"Start of Epoch {epoch+1}")
        # total_patches = sum(SlideDataset(Path(train_fv_path / f"{slide_id}.zarr"), cohort, task, batch_size=batch_size).num_batches for slide_id, _ in train_dataset_all_bags)
        total_train_loss = 0
        train_accuracy.reset()
        train_auroc.reset()
        train_precision.reset()
        train_recall.reset()
        # train_confusion_matrix.reset()
        train_cm = None

        #storage for confusion matrix
        train_labels_list = []
        train_preds_list = []

        with tqdm(total=wsi_count, desc=f"Epoch {epoch+1}/{num_epochs}", unit='WSI') as pbar:
            for slide_id, label in train_dataset_all_bags:
                slide_path = train_fv_path / f"{slide_id}.zarr"

                # print(f"About to use slide_path: {slide_path}")

                if slide_path.exists():
                    slide_dataset = SlideDataset(slide=slide_path, cohort=cohort, task=task, batch_size=batch_size)

                    # Load all patches of the WSI
                    features, coords = next(iter(slide_dataset))

                    #print info on features and coords so I can understand what they look like and later create a dummy batch based on the info. Need all dim info and data type info
                    logging.debug(f"Features shape: {features.shape}, Features dtype: {features.dtype}")
                    logging.debug(f"Coords shape: {coords.shape}, Coords dtype: {coords.dtype}")
                    #print small sample of features and coords
                    logging.debug(f"Sample of features: {features[:5]}")
                    logging.debug(f"Sample of coords: {coords[:5]}")

                    logging.debug(f"Info on label: {label}")
                    logging.debug(f"Type of label: {type(label)}")
                    logging.debug(f"Label shape: {label.shape}")
                    logging.debug(f"Label dtype: {label.dtype}")

                    mem_before = get_cached_memory_in_gb()
                    print_gpu_memory_stats(f"Before processing {slide_id}")
                    assert features.shape[0] == slide_dataset.num_patches, f"Mismatch in number of patches for {slide_id}"
                    features, labels = features.to(device), torch.tensor([label], dtype=torch.float).to(device)

                    # Zero the parameter gradients
                    optimiser.zero_grad()

                    # Forward pass with autocast
                    with autocast(enabled=use_amp):
                        loss, labels, probs, preds = process_batch(features, labels)

                    # Backward and optimize
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimiser)
                        scaler.update()
                    else:
                        loss.backward()
                        optimiser.step()

                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item(), 'WSI': slide_id})


                    total_train_loss += loss.item()
                    train_accuracy.update(preds, labels)
                    train_auroc.update(probs, labels)
                    train_precision.update(preds, labels)
                    train_recall.update(preds, labels)
                    train_labels_list.append(labels.float().cpu().detach().numpy().flatten())
                    train_preds_list.append(preds.float().cpu().detach().numpy().flatten())

                    mem_after = get_cached_memory_in_gb()
                    print_gpu_memory_stats(f"After processing {slide_id}")
                    # Check and log if memory increase is greater than 5GB
                    logging.debug("Attempting to log memory increase...")
                    memory_increase = mem_after - mem_before
                    if memory_increase > 5:  # You might lower this for testing
                        logging.debug(f"{slide_id}: Memory increased by {memory_increase:.2f} GB\n")
                        logging.debug("Logged memory increase.")
                    else:
                        logging.debug("No significant memory increase to log.")

                else:
                    print(f"Slide data for {slide_id} not found in {train_fv_path}\n Slide Path: {slide_path}")
        
        avg_train_loss = total_train_loss / len(train_dataset_all_bags)
        train_acc = train_accuracy.compute()
        train_auc = train_auroc.compute()
        train_prec = train_precision.compute()
        train_rec = train_recall.compute()

        flat_train_labels = np.concatenate(train_labels_list).ravel()
        flat_train_preds = np.concatenate(train_preds_list).ravel()

        counts, values = np.unique(flat_train_labels, return_counts=True)
        # print(f"Unique values in train labels:\n{values}")
        # print(f"Counts of unique values in train labels:\n{counts}")

        counts, values = np.unique(flat_train_preds, return_counts=True)
        # print(f"Unique values in train preds:\n{values}")
        # print(f"Counts of unique values in train preds:\n{counts}")

        # Apply binary thresholding to convert probabilities to binary predictions
        binary_train_preds = (flat_train_preds > 0.5).astype(int)
        train_cm_title = f"Validation CM, Epoch: {epoch+1}/{num_epochs}"
        train_cm = wandb.plot.confusion_matrix(title=train_cm_title, probs=None, y_true=flat_train_labels, preds=binary_train_preds, class_names=get_class_names_from_label_encodings(cohort, task))


        # Validation loop at the end of each epoch
        transformer_model.eval()  # Switch to evaluation mode

        val_accuracy.reset()
        val_auroc.reset()
        val_precision.reset()
        val_recall.reset()
        # val_confusion_matrix.reset()
        val_cm = None
        total_val_loss = 0

        #storage for confusion matrix
        val_labels_list = []
        val_preds_list = []

        with torch.no_grad(), tqdm(total=len(val_dataset_all_bags), desc=f"Validating Epoch {epoch+1}/{num_epochs}", unit='WSI') as pbar:
            for slide_id, label in val_dataset_all_bags:
                slide_path = val_fv_path / f"{slide_id}.zarr"
                if slide_path.exists():
                    slide_dataset = SlideDataset(slide=slide_path, cohort=cohort, task=task, batch_size=batch_size)
                    features, _ = next(iter(slide_dataset))
                    mem_before = get_cached_memory_in_gb()
                    print_gpu_memory_stats(f"Validation - Before processing {slide_id}")
                    features, labels = features.to(device), torch.tensor([label], dtype=torch.float).to(device)

                    # Forward pass with autocast for AMP
                    with autocast(enabled=use_amp):
                        loss, labels, probs, preds = process_batch(features, labels)
                    
                    mem_after = get_cached_memory_in_gb()
                    print_gpu_memory_stats(f"Validation - After processing {slide_id}")
                    memory_increase = mem_after - mem_before
                    if memory_increase > 5:  # You might lower this for testing
                        with open(log_path, "a") as log_file:
                            log_file.write(f"Validation - {slide_id}: Memory increased by {memory_increase:.2f} GB\n")

                    # Update validation metrics
                    total_val_loss += loss.item()
                    val_accuracy.update(preds, labels)
                    val_auroc.update(probs, labels)
                    val_precision.update(preds, labels)
                    val_recall.update(preds, labels)
                    val_labels_list.append(labels.float().cpu().detach().numpy().flatten())
                    val_preds_list.append(preds.float().cpu().detach().numpy().flatten())

                    # Update the progress bar
                    pbar.update(1)
                    # pbar.set_postfix({'val_loss': loss.item()})
                    pbar.set_postfix({'loss': loss.item(), 'WSI': slide_id})

                else:
                    print(f"Validation slide data for {slide_id} not found in {val_fv_path}\n Slide Path: {slide_path}")

        avg_val_loss = total_val_loss / len(val_dataset_all_bags)
        val_acc = val_accuracy.compute().item()
        val_auc = val_auroc.compute().item()
        val_prec = val_precision.compute().item()
        val_rec = val_recall.compute().item()
        #create wandb cm and use class names from task and cohort to get keys for confusion matrix

        flatten_val_labels = np.concatenate(val_labels_list).ravel()
        flatten_val_preds = np.concatenate(val_preds_list).ravel()

        #print labels vs preds for sanity check
        # print(f"Val labels: {flatten_val_labels}")
        # print(f"Val preds: {flatten_val_preds}")

        #print the counts of each of the unique labels and predictions
        values, counts = np.unique(flatten_val_labels, return_counts=True)
        # print(f"Unique values in val labels:\n{values}")
        # print(f"Counts of unique values in val labels:\n{counts}")

        values, counts = np.unique(flatten_val_preds, return_counts=True)
        # print(f"Unique values in val preds:\n{values}")
        # print(f"Counts of unique values in val preds:\n{counts}")

        # Apply binary thresholding to convert probabilities to binary predictions
        binary_val_preds = (flatten_val_preds > 0.5).astype(int)

        val_cm_title = f"Validation CM, Epoch: {epoch+1}/{num_epochs}"
        val_cm = wandb.plot.confusion_matrix( title=val_cm_title, probs=None, y_true=flatten_val_labels, preds=binary_val_preds, class_names=get_class_names_from_label_encodings(cohort, task))

        current_metrics = {
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_auroc": val_auc,
        }

        for metric_name, metric_value in current_metrics.items():
            is_better = metric_value < best_metrics[metric_name]["value"] if metric_name == "val_loss" else metric_value > best_metrics[metric_name]["value"]
            if is_better:
                best_metrics[metric_name]["value"] = metric_value
                best_metrics[metric_name]["epoch"] = epoch
                wandb.run.summary[f"best_{metric_name}"] = metric_value
                wandb.run.summary[f"best_{metric_name}_epoch"] = epoch

                #If the current val_loss is the best, save the model
                try:
                    if metric_name == "val_loss":
                        delete_previous_best_models(run_dir, "val_loss")
                        model_path_str_loss = str(model_save_path_loss).format(epoch=epoch, loss=avg_val_loss)  # Convert to string and format
                        torch.save(transformer_model.state_dict(), model_path_str_loss)
                        print(f"New best model will be saved to {model_path_str_loss}")
                    elif metric_name == "val_auroc":
                        delete_previous_best_models(run_dir, "val_auroc")
                        model_path_str_auroc = str(model_save_path_auroc).format(epoch=epoch, auroc=val_auc)  # Convert to string and format
                        torch.save(transformer_model.state_dict(), model_path_str_auroc)
                        print(f"New best model will be saved to {model_path_str_auroc}")
                except IOError as e:
                    print(f"Failed to save the model due to IOError: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while saving the model: {e}")

        # Log metrics with W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "train_auroc": train_auc,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "val_auroc": val_auc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "train_confusion_matrix": train_cm,
            "val_confusion_matrix": val_cm
        })

        # Print some metrics
        print(f"Training Metrics: \tEpoch {epoch+1}/{num_epochs}, Avg. Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}, AUROC: {train_auc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        print(f"Validation Metrics: \tEpoch {epoch+1}/{num_epochs}, Avg. Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}, AUROC: {val_auc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")

        # Switch back to training mode for the next epoch
        transformer_model.train()

        # Next epoch loop

def get_class_names_from_label_encodings(cohort, task):
    # Extract the specific label encoding mapping for the task and cohort
    label_mapping = LABEL_ENCODINGS[cohort][task]
    
    # Sort the label mapping by its values (0, 1) to ensure the correct order
    sorted_label_mapping = sorted(label_mapping.items(), key=lambda x: x[1])
    
    # Extract the class names in the sorted order
    class_names = [label for label, _ in sorted_label_mapping]
    
    return class_names

if __name__ == "__main__":
    main()
