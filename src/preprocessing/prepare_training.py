import json
import random
import logging
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.youtube import process_playlists
from constants import *
from utils.aws import save_json_to_s3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_train_val_split(video_metadata: Dict[str, Dict], val_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
    """
    Generate train-validation splits by video.

    Args:
        video_metadata (Dict[str, Dict]): Dictionary of video metadata.
        val_ratio (float): Ratio of videos to use for validation. Default is 0.2.

    Returns:
        Tuple[List[str], List[str]]: Lists of video IDs for training and validation sets.
    """
    # Get video IDs
    video_ids = list(video_metadata.keys())
    random.shuffle(video_ids)
    
    # Split into train and validation sets
    split_index = int(len(video_ids) * (1 - val_ratio))
    train_ids = video_ids[:split_index]
    val_ids = video_ids[split_index:]
    
    return train_ids, val_ids

def main():
    # Load channels from JSON file
    with open(CHANNELS_FILENAME, 'r') as f:
        channels = json.load(f)

    # Process channels
    run_name = input("Enter run name: ")
    video_metadata = process_playlists(channels, mode='train', n=N_SENTENCES, run_name=run_name)
    logger.info(f"Processed {len(video_metadata)} videos.")

    # Generate train-val split
    train_ids, val_ids = generate_train_val_split(video_metadata, SPLIT_HOLD_OUT_RATIO)
    logger.info(f"Train set: {len(train_ids)} videos")
    logger.info(f"Validation set: {len(val_ids)} videos")

    # Save train and validation splits
    splits = {
        "train": train_ids,
        "validation": val_ids
    }
    s3_file_key = f"{S3_DATA_DIR}/{str(run_name)}/{SPLITS_FILENAME}"
    save_json_to_s3(splits, S3_BUCKET_NAME, s3_file_key)
    
    logger.info("Train-validation splits saved to S3")

if __name__ == "__main__":
    main()
