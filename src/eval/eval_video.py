import os
import sys
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.train import PodcastSegmentationModel
from utils.youtube import create_youtube_client, get_podcast_details, get_podcast_id_from_url
from utils.aws import read_json, download_file_from_s3, save_json_to_s3
from constants import *

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(checkpoint_path, device, model_type, input_dim, hidden_dim, num_layers, num_heads=None):
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

def run_inference(model, sentence_embeddings, device):
    with torch.no_grad():
        sentence_embeddings = torch.tensor(sentence_embeddings).float().to(device)
        logits = model(sentence_embeddings.unsqueeze(0))
        predictions = torch.sigmoid(logits).cpu().numpy()
    return predictions[0].flatten()

def main():
    # Set up device (CUDA, MPS, or CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Get input from user
    video_url = input("Enter the YouTube video URL: ")
    config_name = input("Enter the config name: ")

    # Extract video ID from URL
    try:
        video_id = get_podcast_id_from_url(video_url)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

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

    # Create YouTube client and get podcast details
    youtube_client = create_youtube_client()
    podcast_details = get_podcast_details(youtube_client, video_id, mode='inference', n=4) #config['num_sentences'])
    if podcast_details is None:
        print(f"Error: Unable to retrieve details for video {video_id}")
        sys.exit(1)

    # Run inference
    predictions = run_inference(model, podcast_details['sentence_embeddings'], device)

    # Create binary predictions
    binary_predictions = (predictions > model.segment_threshold).astype(int)

    # Prepare results
    results = {
        'video_id': video_id,
        'title': podcast_details['title'],
        'sentences': podcast_details['sentences'],
        'sentence_start_times': podcast_details['sentence_start_times'],
        'segments': podcast_details['segments'],
        'segment_start_times': podcast_details['segment_start_times'],
        'ground_truths': podcast_details['segment_indicators'],
        'predictions': binary_predictions.tolist(),
        'raw_predictions': predictions.tolist()
    }

    # Save results to S3
    s3_file_key = f"{S3_MODELS_DIR}/eval/{video_id}_results.json"
    
    try:
        save_json_to_s3(results, S3_BUCKET_NAME, s3_file_key)
        print(f"Results saved to S3: s3://{S3_BUCKET_NAME}/{s3_file_key}")
    except Exception as e:
        print(f"Error saving results to S3: {str(e)}")
        print("Results were not saved.")

    # Print a summary of the results
    print("\nInference Results Summary:")
    print(f"Video ID: {video_id}")
    print(f"Title: {podcast_details['title']}")
    print(f"Number of sentences: {len(podcast_details['sentences'])}")
    print(f"Number of predicted segments: {sum(binary_predictions)}")
    print(f"Results saved to: s3://{S3_BUCKET_NAME}/{s3_file_key}")

if __name__ == "__main__":
    main()
