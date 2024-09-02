import os
import sys
import torch
import numpy as np
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.train import PodcastSegmentationModel
from modeling.dataloader import VideoDataModule
from utils.metrics import windiff
from utils.aws import save_json_to_s3, read_json, download_file_from_s3
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

def load_model(checkpoint_path, device, model_type, input_dim, hidden_dim, num_layers, num_heads=None):
    """
    Load the pre-trained model from a checkpoint and move it to the specified device.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        device (torch.device): The device to load the model onto.
        model_type (str): Type of model ('bilstm' or 'transformer').
        input_dim (int): Input dimension for the model.
        hidden_dim (int): Hidden dimension for the model.
        num_layers (int): Number of layers in the model.
        num_heads (int, optional): Number of attention heads (for transformer model).

    Returns:
        PodcastSegmentationModel: The loaded model in evaluation mode.
    """
    model_kwargs = {
        "model_type": model_type,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    
    if model_type == 'transformer' and num_heads is not None:
        model_kwargs["num_heads"] = num_heads

    model = PodcastSegmentationModel.load_from_checkpoint(
        checkpoint_path,
        **model_kwargs
    )
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_module, device):
    """
    Evaluate the model on the validation set and save results at the video_id level.

    Args:
        model (PodcastSegmentationModel): The model to evaluate.
        data_module (VideoDataModule): The data module containing the validation set.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: A tuple containing:
            - float: The computed WinDiff score for the entire validation set.
            - dict: A dictionary with results for each video_id.
    """
    data_module.setup(stage='validate')
    val_dataloader = data_module.val_dataloader()

    results_by_video = {}
    all_predictions = []
    all_ground_truths = []

    # Iterate through the validation set
    with torch.no_grad():
        for batch in val_dataloader:
            sentence_embeddings = batch['sentence_embeddings'].to(device)
            segment_indicators = batch['segment_indicators'].cpu().numpy()
            video_ids = batch['video_id']

            # Get model predictions
            logits = model(sentence_embeddings)
            predictions = torch.sigmoid(logits).cpu().numpy()

            # Process results for each video in the batch
            for i, video_id in enumerate(video_ids):
                pred = predictions[i].flatten()
                true = segment_indicators[i].flatten()
                
                pred_binary = (pred > model.segment_threshold).astype(int)
                true_binary = true.astype(int)
                
                windiff_score = windiff(true_binary, pred_binary, model.window_size)
                
                results_by_video[video_id] = {
                    "predictions": pred_binary.tolist(),
                    "ground_truths": true_binary.tolist(),
                    "windiff_score": windiff_score
                }

                # Accumulate predictions and ground truths for overall score
                all_predictions.extend(pred_binary)
                all_ground_truths.extend(true_binary)

    # Compute overall WinDiff score
    overall_windiff = windiff(np.array(all_ground_truths), np.array(all_predictions), model.window_size)

    # Add overall score to results
    results_by_video["overall"] = {"windiff_score": overall_windiff}

    return overall_windiff, results_by_video

if __name__ == "__main__":
    # Set up device (CUDA, MPS, or CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Get config name from user
    config_name = input("Enter the config name: ")

    # Load config from S3
    s3_config_key = f"{S3_MODELS_DIR}/{config_name}/config.json"
    try:
        config = read_json(S3_BUCKET_NAME, s3_config_key)
        print("Config loaded successfully from S3.")
    except Exception as e:
        print(f"Error loading config from S3: {str(e)}")
        sys.exit(1)

    # Download model checkpoint from S3
    s3_checkpoint_key = f"{S3_MODELS_DIR}/{config_name}/best_model.ckpt"
    local_checkpoint_path = f"./tmp_{config_name}_best_model.ckpt"
    try:
        download_file_from_s3(S3_BUCKET_NAME, s3_checkpoint_key, local_checkpoint_path)
        print("Model checkpoint downloaded successfully from S3.")
    except Exception as e:
        print(f"Error downloading model checkpoint from S3: {str(e)}")
        sys.exit(1)

    # Load the model
    model_type = config['model_type']
    input_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    num_layers = config['num_layers']
    num_heads = config['num_heads'] if model_type == 'transformer' else None
    model = load_model(local_checkpoint_path, device, model_type, input_dim, hidden_dim, num_layers, num_heads)
    print("Model loaded successfully.")

    # Clean up the temporary checkpoint file
    os.remove(local_checkpoint_path)

    # Set up data module with stride equal to sequence length
    preproc_run_name = config['train_data_folder']
    seq_length = config['seq_length']
    data_module = VideoDataModule(preproc_run_name, batch_size=1, seq_length=seq_length, stride=seq_length)
    print(f"Data module initialized with sequence length and stride of {seq_length}.")

    # Evaluate the model
    print("Starting model evaluation...")
    overall_windiff, results_by_video = evaluate_model(model, data_module, device)

    # Print results
    print("\nEvaluation Results:")
    print(f"Overall WinDiff: {overall_windiff:.4f}")
    print("Lower WinDiff scores indicate better performance.")

    # Save results to S3
    s3_file_key = f"{S3_MODELS_DIR}/eval/{preproc_run_name}_results.json"
    
    try:
        save_json_to_s3(results_by_video, S3_BUCKET_NAME, s3_file_key)
        print(f"Results saved to S3: s3://{S3_BUCKET_NAME}/{s3_file_key}")
    except Exception as e:
        print(f"Error saving results to S3: {str(e)}")
        print("Results were not saved.")
