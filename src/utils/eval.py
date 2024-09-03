import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.train import PodcastSegmentationModel
from utils.aws import download_file_from_s3
from constants import *

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

def load_model_from_config(config, config_name, device):
    """
    Load the model based on the given configuration name.

    Args:
        config_name (str): Name of the configuration to use.
        device (torch.device): The device to load the model onto.

    Returns:
        tuple: A tuple containing:
            - PodcastSegmentationModel: The loaded model.
            - dict: The configuration dictionary.
    """
    # Download model checkpoint from S3
    s3_checkpoint_key = f"{S3_MODELS_DIR}/{config_name}/best_model.ckpt"
    local_checkpoint_path = f"./tmp_{config_name}_best_model.ckpt"
    try:
        download_file_from_s3(S3_BUCKET_NAME, s3_checkpoint_key, local_checkpoint_path)
        print("Model checkpoint downloaded successfully from S3.")
    except Exception as e:
        print(f"Error downloading model checkpoint from S3: {str(e)}")
        raise

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

    return model

def plot_predictions_vs_ground_truths(pred_dict, ground_truths, video_id):
    """Plot predictions vs ground truths for multiple models."""
    num_models = len(pred_dict)
    fig, axes = plt.subplots(num_models + 1, 1, figsize=(8, 4 * (num_models + 1)))

    # Plot ground truths first
    axes[0].plot(ground_truths, color='red')
    axes[0].set_title(f'Ground Truths: {video_id}')
    axes[0].set_ylabel('Activation')
    axes[0].set_ylim(0, 1)

    # Plot predictions for each model
    for i, (model_name, vals) in enumerate(pred_dict.items(), start=1):
        predictions = vals["predictions"]
        axes[i].plot(predictions, color='blue')
        axes[i].set_title(f'Predictions ({model_name}): {video_id}')
        axes[i].set_ylabel('Activation')
        axes[i].set_ylim(0, 1)

    axes[-1].set_xlabel('Sentence Index')

    plt.tight_layout()
    plt.show()

# Plot distribution of predictions array
def plot_prediction_distribution(predictions):
    """Plot distribution of predictions array."""
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50, edgecolor='black')
    plt.title('Distribution of Predictions')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.show()

def plot_mean_windiff(model_names, mean_windiff_values, ax):
    ax.bar(model_names, mean_windiff_values)
    ax.set_title('Mean WinDiff Values by Model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean WinDiff')
    ax.set_ylim(0, max(mean_windiff_values) * 1.1)
    for i, v in enumerate(mean_windiff_values):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')

def plot_mean_topic_diff(model_names, mean_topic_diff_values, ax):
    ax.bar(model_names, mean_topic_diff_values)
    ax.set_title('Mean Topic Diff Values by Model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean Topic Diff')
    ax.set_ylim(min(mean_topic_diff_values) * 1.1, max(mean_topic_diff_values) * 1.1)
    for i, v in enumerate(mean_topic_diff_values):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')