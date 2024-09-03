import os
import sys
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.llm_inference import llm_inference
from utils.youtube import create_youtube_client, get_podcast_details, get_podcast_id_from_url
from utils.aws import read_json, save_json_to_s3
from utils.device import get_device
from utils.eval import load_model_from_config
from constants import *
from utils.metrics import windiff


def run_ml_inference(model, sentence_embeddings, device):
    seq_length = model.seq_length if hasattr(model, 'seq_length') else 256

    with torch.no_grad():
        sentence_embeddings = torch.tensor(sentence_embeddings).float().to(device)
        
        if len(sentence_embeddings) <= seq_length:
            # If embeddings fit within sequence length, process normally
            logits = model(sentence_embeddings.unsqueeze(0))
            predictions = torch.sigmoid(logits).cpu().numpy()[0]
        else:
            # Apply chunking for longer sequences
            chunks = []
            for i in range(0, len(sentence_embeddings) - seq_length + 1, seq_length):
                chunk = sentence_embeddings[i:i+seq_length]
                logits = model(chunk.unsqueeze(0))
                chunks.append(torch.sigmoid(logits).cpu().numpy()[0])
            
            # Combine chunks with sliding inference window
            predictions = np.zeros(len(sentence_embeddings))
            for i, chunk in enumerate(chunks):
                start = i * seq_length
                end = start + seq_length
                if i > 0:
                    chunk[0] = 0
                predictions[start:end] = chunk.flatten()

            # Trim to original length
            predictions = predictions[:len(sentence_embeddings)].flatten()
            pred_binary = (predictions > model.segment_threshold).astype(int)

    return pred_binary, predictions

def main(video_url, config_name):
    # Extract video ID from URL
    try:
        video_id = get_podcast_id_from_url(video_url)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    # Set up device (CUDA, MPS, or CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Load config from S3
    s3_config_key = f"{S3_MODELS_DIR}/{config_name}/config.json"
    config = read_json(S3_BUCKET_NAME, s3_config_key)

    # Load the model and config
    model_type = config.get("model_type")
    if model_type in ["transformer", "lstm"]:
        model = load_model_from_config(config, config_name, device)
        llm_kwargs = None
    else:
        model = config["model_name"]
        llm_kwargs = config["model_kwargs"]

    # Create YouTube client and get podcast details
    youtube_client = create_youtube_client()
    podcast_details, _ = get_podcast_details(youtube_client, video_id, mode='inference', n_chunks=8)#config['n_chunks'])
    if podcast_details is None:
        print(f"Error: Unable to retrieve details for video {video_id}")
        sys.exit(1)

    # Run inference
    model_type = config.get("model_type")
    if model_type in ["transformer", "lstm"]:
        binary_predictions, raw_predictions = run_ml_inference(model, podcast_details['sentence_embeddings'], device)
        actual_transitions = podcast_details['segment_indicators']
        binary_predictions = binary_predictions.tolist()
    else:
        binary_predictions, actual_transitions, raw_predictions = llm_inference(video_id, model, **llm_kwargs)

    # Calculate metrics
    windiff_val = windiff(actual_transitions, binary_predictions, WINDOW_SIZE)

    # Prepare results
    results = {
        'video_id': video_id,
        'title': podcast_details['title'],
        'sentences': podcast_details['sentences'],
        'sentence_start_times': podcast_details['sentence_start_times'],
        'segments': podcast_details['segments'],
        'ground_truths': actual_transitions,
        'predictions': binary_predictions,
        'raw_predictions': raw_predictions.tolist() if isinstance(raw_predictions, np.ndarray) else raw_predictions
    }

    # Save results to S3
    s3_file_key = f"{S3_MODELS_DIR}/{config_name}/eval/{video_id}_results.json"
    save_json_to_s3(results, S3_BUCKET_NAME, s3_file_key)

    # Print a summary of the results
    print("\nInference Results Summary:")
    print(f"Video ID: {video_id}")
    print(f"Title: {podcast_details['title']}")
    print(f"Number of actual segments: {len(podcast_details['segments'])}")
    print(f"Number of predicted segments: {sum(binary_predictions)}")
    print(f"WindowDiff: {windiff_val}")
    print(f"Results saved to: s3://{S3_BUCKET_NAME}/{s3_file_key}")

if __name__ == "__main__":
    # Get input from user
    video_url = "https://www.youtube.com/watch?v=tdv7r2JSokI" #input("Enter the YouTube video URL: ")
    config_name = input("Enter the config name: ")
    main(video_url, config_name)
