import os
import sys
import torch
import numpy as np
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.train import PodcastSegmentationModel
from modeling.dataloader import VideoDataModule
from utils.metrics import windiff
from utils.aws import save_json_to_s3
from constants import *

def get_device():
    """
    Determine the best available device for computation.

    Returns:
        torch.device: The selected device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(checkpoint_path, device, input_dim, hidden_dim, num_layers):
    """
    Load the pre-trained model from a checkpoint and move it to the specified device.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        device (torch.device): The device to load the model onto.
        input_dim (int): Input dimension for the model.
        hidden_dim (int): Hidden dimension for the model.
        num_layers (int): Number of layers in the model.

    Returns:
        PodcastSegmentationModel: The loaded model in evaluation mode.
    """
    model = PodcastSegmentationModel.load_from_checkpoint(
        checkpoint_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_module, device):
    """
    Evaluate the model on the validation set.

    This function processes the entire validation set, making predictions
    for each sequence and computing the WinDiff metric for the whole dataset.

    Args:
        model (PodcastSegmentationModel): The model to evaluate.
        data_module (VideoDataModule): The data module containing the validation set.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: A tuple containing:
            - float: The computed WinDiff score for the entire validation set.
            - dict: A dictionary with prediction labels and actual labels.
    """
    data_module.setup(stage='validate')
    val_dataloader = data_module.val_dataloader()

    all_predictions = []
    all_ground_truths = []

    # Iterate through the validation set
    with torch.no_grad():
        for batch in val_dataloader:
            sentence_embeddings = batch['sentence_embeddings'].to(device)
            segment_indicators = batch['segment_indicators'].cpu().numpy()

            # Get model predictions
            logits = model(sentence_embeddings)
            predictions = torch.sigmoid(logits).cpu().numpy()

            # Accumulate predictions and ground truths
            all_predictions.extend(predictions)
            all_ground_truths.extend(segment_indicators)

    # Flatten predictions and ground truths
    full_predictions = np.concatenate([pred.flatten() for pred in all_predictions])
    full_ground_truths = np.concatenate([gt.flatten() for gt in all_ground_truths])

    # Binarize predictions
    pred_binary = (full_predictions > model.segment_threshold).astype(int)
    true_binary = full_ground_truths.astype(int)

    # Compute WinDiff
    windiff_score = windiff(true_binary, pred_binary, model.window_size)

    # Prepare results dictionary
    results = {
        "predictions": pred_binary.tolist(),
        "ground_truths": true_binary.tolist(),
        "windiff_score": windiff_score
    }

    return windiff_score, results

if __name__ == "__main__":
    # Set up device (CUDA, MPS, or CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Load the model
    checkpoint_path = "/Users/johncalzaretta/Desktop/projects/nlp_podcast_segmentation/src/modeling/checkpoints/podcast_segmentation-epoch=39-val_loss=0.49.ckpt" #input("Enter the path to the model checkpoint: ")
    
    # Model parameters (you might need to adjust these based on your actual model configuration)
    input_dim = 384  # This should match your sentence embedding dimension
    hidden_dim = 256  # This should match your model's hidden dimension
    num_layers = 2   # This should match your model's number of layers
    
    model = load_model(checkpoint_path, device, input_dim, hidden_dim, num_layers)
    print("Model loaded successfully.")

    # Set up data module with stride equal to sequence length
    preproc_run_name = "lex_10" #input("Enter the preprocessing run name: ")
    seq_length = 512  # Adjust this value if your sequence length is different
    data_module = VideoDataModule(preproc_run_name, batch_size=1, seq_length=seq_length, stride=seq_length)
    print(f"Data module initialized with sequence length and stride of {seq_length}.")

    # Evaluate the model
    print("Starting model evaluation...")
    windiff_score, results = evaluate_model(model, data_module, device)

    # Print results
    print("\nEvaluation Results:")
    print(f"WinDiff: {windiff_score:.4f}")
    print("Lower WinDiff scores indicate better performance.")

    # Save results to S3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_file_key = f"{S3_DATA_DIR}/{preproc_run_name}/eval/evaluation_results_{timestamp}.json"
    
    try:
        save_json_to_s3(results, S3_BUCKET_NAME, s3_file_key)
        print(f"Results saved to S3: s3://{S3_BUCKET_NAME}/{s3_file_key}")
    except Exception as e:
        print(f"Error saving results to S3: {str(e)}")
        print("Results were not saved.")
