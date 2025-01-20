import torch
import argparse
from pathlib import Path
from datasets.dataset_constants import TASKS, AVAILABLE_COHORTS, LABEL_ENCODINGS
from datasets.dataset_loader import Dataset_All_Bags, SlideDataset
import torchmetrics
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import os
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

# Assuming other necessary imports are done earlier in the script

class TransformerForClassification(nn.Module):
    def __init__(self, fv_extractor_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="gelu", layer_norm_eps=1e-5):
        super(TransformerForClassification, self).__init__()
        self.fc = nn.Linear(fv_extractor_size, d_model)  # Fully connected layer to transform input features from 768 (ctranspath dim ,for example) to d_model
        self.relu = nn.ReLU()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation, layer_norm_eps=layer_norm_eps)
        self.classifier = nn.Linear(d_model, 1)  # Assuming binary classification

    def forward(self, src):
        # src expected shape: (sequence_length, batch_size, feature vector dim)
        src = self.relu(self.fc(src))  # Apply the fully connected layer and ReLU
        memory = self.transformer.encoder(src)  # Now src is of shape (sequence_length, batch_size, d_model)
        out = self.classifier(memory.mean(dim=0))  # Mean pooling across the sequence_length dimension
        return out.view(-1)  # Reshape the output to match the target tensor shape

def get_class_names_from_label_encodings(cohort, task):
    # Extract the specific label encoding mapping for the task and cohort
    label_mapping = LABEL_ENCODINGS[cohort][task]
    
    # Sort the label mapping by its values (0, 1) to ensure the correct order
    sorted_label_mapping = sorted(label_mapping.items(), key=lambda x: x[1])
    
    # Extract the class names in the sorted order
    class_names = [label for label, _ in sorted_label_mapping]
    
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained transformer model.')

    # Model configuration parameters
    parser.add_argument('--d_model', type=int, default=512, help='number of expected features in transformer')
    parser.add_argument('--encoder_layers', type=int, help='number of encoder layers', default=6)
    parser.add_argument('--nhead', type=int, help='number of transformer heads. Default 4 but torch default suggests 8.', default=4)
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of the feedforward network model')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--activation', type=str, default="gelu", help='activation function')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5, help='layer normalization epsilon')
    parser.add_argument('--feature_extractor', type=str, default="ctranspath", choices=["ctranspath", "owkin", "resnet50", "resnet50-b", "uni"], help='feature extractor to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducibility')

    # Evaluation specific parameters
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model file')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification')
    parser.add_argument('--test_fv_path', type=str, required=True, help='path to test feature vectors (zarr)')
    parser.add_argument('--test_dataset_csv_path', type=str, required=True, help='path to test dataset CSV')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--task', type=str, help='task to train classifier for')
    parser.add_argument('--cohort', type=str, help='cohort to train classifier for', choices=AVAILABLE_COHORTS)
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision')

    args = parser.parse_args()

    # Validate the task for the selected cohort
    valid_tasks_for_cohort = TASKS.get(args.cohort, [])
    if args.task not in valid_tasks_for_cohort:
        valid_tasks_str = ', '.join(valid_tasks_for_cohort)
        parser.error(f"The task '{args.task}' is not valid for cohort '{args.cohort}'. "
                     f"Valid tasks for this cohort are: {valid_tasks_str}")

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_fv_path = Path(args.test_fv_path)
    test_fv_path = test_fv_path / args.feature_extractor

    # Feature extractor size needs to be determined based on the feature_extractor argument
    if args.feature_extractor in ["ctranspath", "owkin"]:
        ft_extractor_size = 768
    elif args.feature_extractor in ["resnet50-b", "uni"]:
        ft_extractor_size = 1024
    elif args.feature_extractor == "resnet50":
        ft_extractor_size = 2048
    else:
        raise ValueError(f"Unknown feature extractor: {args.feature_extractor}")

    print(f"Feature extractor size: {ft_extractor_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #init torch transformer model #! Parameterise these
    d_model = 512 #(default=512). The number of expected features in the encoder/decoder inputs.
    nhead = args.nhead #(default=8). The number of heads in the multiheadattention models.
    num_encoder_layers = args.encoder_layers #(default=6). The number of sub-encoder-layers in the encoder.
    num_decoder_layers = 6 #(default=6). The number of sub-decoder-layers in the decoder.
    dim_feedforward = 2048 #(default=2048). The dimension of the feedforward network model.
    # dropout = 0.1 #(default=0.1). The dropout value.
    dropout = args.dropout #! Parameterised.
    # activation = "gelu" #(default="relu"). he activation function of encoder/decoder intermediate layer, can be a string (“relu” or “gelu”) or a unary callable.
    activation = args.activation #! Parameterised.
    custom_encoder = None #(default=None).
    custom_decoder = None #(default=None).
    layer_norm_eps = 1e-5 #(default=1e-5). The eps value in layer normalization components. (add epsilon to the denominator to avoid division by zero)
    batch_first = False #(default=False) If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
    norm_first = False #(default=False) If True, encoder and decoder layers will perform LayerNorms before other attention and feedforward operations, otherwise after. Default: False (after).

    # Initialize and load the trained model
    model = TransformerForClassification(
        fv_extractor_size=ft_extractor_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Load the test dataset
    test_dataset_all_bags = Dataset_All_Bags(csv_path=args.test_dataset_csv_path, task=args.task, cohort=args.cohort, shuffle=False)
    num_classes = len(LABEL_ENCODINGS[args.cohort][args.task])

    # Evaluate the model
    model.eval()  # Switch to evaluation mode

    # Determine the task type based on the number of classes
    task_type = "binary" if num_classes == 2 else "multiclass"

    # Initialize storing lists for outputs and labels
    all_outputs = []
    all_labels = []

    # Initialize metrics with the 'task' parameter
    test_accuracy = torchmetrics.Accuracy(task=task_type, num_classes=num_classes).to(device)
    test_auroc = torchmetrics.AUROC(task=task_type, num_classes=num_classes).to(device)
    test_precision = torchmetrics.Precision(task=task_type, num_classes=num_classes).to(device)
    test_recall = torchmetrics.Recall(task=task_type, num_classes=num_classes).to(device)
    test_confusion_matrix = torchmetrics.ConfusionMatrix(task=task_type, num_classes=num_classes).to(device)

    with torch.no_grad(), tqdm(total=len(test_dataset_all_bags), desc="Evaluating", unit='WSI') as pbar:
        for slide_id, label in test_dataset_all_bags:
            slide_path = test_fv_path / f"{slide_id}.zarr"
            if slide_path.exists():
                slide_dataset = SlideDataset(slide=slide_path, cohort=args.cohort, task=args.task, batch_size=args.batch_size)
                # Load all patches of the WSI
                features, coords = next(iter(slide_dataset))
                assert features.shape[0] == slide_dataset.num_patches, f"Mismatch in number of patches for {slide_id}"
                features, labels = features.to(device), torch.tensor([label], dtype=torch.float).to(device)

                with autocast(enabled=args.use_amp):
                    outputs = model(features)

                #print the logit and label
                # print(f"Logits: {outputs}, Label: {labels}")
                #print the sigmoid output and label
                # print(f"Sigmoid output: {outputs.sigmoid()}, Label: {labels}")

                all_outputs.extend(outputs.sigmoid().cpu().numpy())  # Apply sigmoid to get probabilities
                all_labels.extend(labels.cpu().numpy())

                # test_accuracy.update(outputs, labels)
                # test_auroc.update(outputs, labels)
                # test_precision.update(outputs, labels)
                # test_recall.update(outputs, labels)
                # test_confusion_matrix.update(outputs.sigmoid().round().int(), labels.int())

                pbar.update(1)

    # Convert the continuous outputs to binary predictions using the provided threshold
    thresholded_predictions = (np.array(all_outputs) >= args.threshold).astype(int)

    # Calculate metrics with the provided threshold
    thresholded_accuracy = accuracy_score(all_labels, thresholded_predictions)
    thresholded_auroc = roc_auc_score(all_labels, all_outputs)  # AUROC is independent of the threshold
    thresholded_precision = precision_score(all_labels, thresholded_predictions)
    thresholded_recall = recall_score(all_labels, thresholded_predictions)
    thresholded_confusion_matrix = confusion_matrix(all_labels, thresholded_predictions)
    
    # Print the metrics after applying the provided threshold
    print(f"\nMetrics using the provided threshold ({args.threshold}):")
    print(f"Accuracy: {thresholded_accuracy}")
    print(f"AUROC: {thresholded_auroc}")
    print(f"Precision: {thresholded_precision}")
    print(f"Recall: {thresholded_recall}")
    print("Confusion Matrix:")
    print(thresholded_confusion_matrix)


if __name__ == "__main__":
    main()