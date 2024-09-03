import os
import sys
import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.train import PodcastSegmentationModel
from modeling.dataloader import VideoDataModule
from modeling.llm_inference import llm_inference
from utils.metrics import windiff
from utils.eval import load_model_from_config
from utils.device import get_device
from utils.aws import save_json_to_s3, read_json, download_file_from_s3
from constants import *


def evaluate_test_set(model, data_module, device, model_type = 'custom', llm_kwargs=None):
    """
    Evaluate the model on the validation set and save results at the video_id level.

    Args:
        model_type (str): 'custom' or 'llm' model
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
        for batch in tqdm(val_dataloader, desc="Evaluating model", total=len(val_dataloader)):
            sentence_embeddings = batch['sentence_embeddings'].to(device)
            segment_indicators = batch['segment_indicators'].cpu().numpy()
            attention_masks = batch['attention_mask'].cpu().numpy()
            video_ids = batch['video_id']

            # Custom model eval
            if model_type == "custom":
                # Get model predictions
                logits = model(sentence_embeddings)
                predictions = torch.sigmoid(logits).cpu().numpy()

                # Process results for each video in the batch
                for i, video_id in enumerate(video_ids):
                    pred = predictions[i].flatten()
                    true = segment_indicators[i].flatten()
                    true_binary = true.astype(int)
                    attention_mask = attention_masks[i].flatten()
                    pred_binary = (pred > model.segment_threshold).astype(int)

                    # Calculate metrics
                    windiff_score = windiff(true_binary, pred_binary, WINDOW_SIZE, attention_mask)
                
                    # Add to results
                    results_by_video[video_id] = {
                        "predictions": pred_binary.tolist(),
                        "ground_truths": true_binary.tolist(),
                        "windiff_score": windiff_score
                    }

                    # Accumulate predictions and ground truths for overall score
                    all_predictions.extend(pred_binary)
                    all_ground_truths.extend(true_binary)
            else:
                # Process results for each video in the batch
                for i, video_id in tqdm(enumerate(video_ids), total=len(video_ids)):
                    pred_binary, true_binary, topic_dict = llm_inference(video_id, model, **llm_kwargs)

                    # Calculate metrics
                    windiff_score = windiff(true_binary, pred_binary, WINDOW_SIZE)
                    topic_diff = sum(true_binary) - sum(pred_binary)

                    # Add to results
                    results_by_video[video_id] = {
                        "predictions": pred_binary,
                        "ground_truths": true_binary,
                        "windiff_score": windiff_score,
                        "topic_diff": topic_diff,
                        "topic_dict": topic_dict
                    }

                    # Accumulate predictions and ground truths for overall score
                    all_predictions.extend(pred_binary)
                    all_ground_truths.extend(true_binary)

    # Compute overall WinDiff score
    overall_windiff = np.mean([result["windiff_score"] for result in results_by_video.values()])
    overall_topic_diff = np.mean([result["topic_diff"] for result in results_by_video.values()])

    # Add overall score to results
    results_by_video["overall"] = {"mean_windiff": overall_windiff, "mean_topic_diff": overall_topic_diff}

    return overall_windiff, results_by_video

def main(config_name):
    """
    Evaluate the model on the test set based on the given configuration.

    Args:
        config_name (str): Name of the configuration to use.

    Returns:
        tuple: A tuple containing:
            - float: The overall WinDiff score.
            - dict: A dictionary with results for each video_id.
    """
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

    # Set up data module with stride equal to sequence length
    preproc_run_name = config['train_data_folder']
    seq_length = config['seq_length']
    data_module = VideoDataModule(preproc_run_name, batch_size=1, seq_length=seq_length, stride=seq_length)
    print(f"Data module initialized with sequence length and stride of {seq_length}.")

    # Evaluate the model
    print("Starting model evaluation...")
    overall_windiff, results_by_video = evaluate_test_set(model, data_module, device, model_type=model_type, llm_kwargs=llm_kwargs)

    # Print results
    print("\nEvaluation Results:")
    print(f"Overall WinDiff: {overall_windiff:.4f}")
    print("Lower WinDiff scores indicate better performance.")

    # Save results to S3
    s3_file_key = f"{S3_MODELS_DIR}/{config_name}/val_results.json"
    
    try:
        save_json_to_s3(results_by_video, S3_BUCKET_NAME, s3_file_key)
        print(f"Results saved to S3: s3://{S3_BUCKET_NAME}/{s3_file_key}")
    except Exception as e:
        print(f"Error saving results to S3: {str(e)}")
        print("Results were not saved.")

    return overall_windiff, results_by_video

if __name__ == "__main__":
    # Get config name from user
    config_name = input("Enter the config name: ")
    
    try:
        overall_windiff, results_by_video = main(config_name)
        print(f"Evaluation completed successfully. Overall WinDiff: {overall_windiff:.4f}")
    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")
        sys.exit(1)
