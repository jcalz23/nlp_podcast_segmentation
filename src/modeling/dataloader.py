import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.aws import read_json
from constants import S3_BUCKET_NAME, S3_DATA_DIR, SPLITS_FILENAME

class VideoDataset(Dataset):
    def __init__(self, video_ids, seq_length=512, stride=256):
        self.video_ids = video_ids
        self.seq_length = seq_length
        self.stride = stride
        self.data = self._load_all_data()
        logging.info(f"Finished loading {len(self.data)} segments from {len(video_ids)} videos")

    def _load_all_data(self):
        all_data = []
        for video_id in self.video_ids:
            data = read_json(S3_BUCKET_NAME, f"{S3_DATA_DIR}/podcasts/{video_id}.json")
            sentence_embedding = torch.tensor(data["sentence_embeddings"])
            segment_inds = torch.tensor(data["segment_indicators"])
            
            # Create multiple records for long podcasts
            for start in range(0, len(sentence_embedding), self.stride):
                end = start + self.seq_length
                if end > len(sentence_embedding):
                    # Pad the last segment if it's shorter than seq_length
                    padding_length = end - len(sentence_embedding)
                    segment = torch.cat([
                        sentence_embedding[start:],
                        torch.zeros(padding_length, sentence_embedding.size(1))
                    ])
                    segment_ind = torch.cat([
                        segment_inds[start:],
                        torch.zeros(padding_length)
                    ])
                    attention_mask = torch.cat([
                        torch.ones(self.seq_length - padding_length),
                        torch.zeros(padding_length)
                    ])
                else:
                    segment = sentence_embedding[start:end]
                    segment_ind = segment_inds[start:end]
                    attention_mask = torch.ones(self.seq_length)
                # Force the first sentence to be a segment start
                segment_ind[0] = 1
                
                all_data.append({
                    "sentence_embeddings": segment,
                    "segment_indicators": segment_ind,
                    "attention_mask": attention_mask,
                    "video_id": video_id
                })
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, preproc_run_name, batch_size=32, seq_length=512, stride=256):
        super().__init__()
        self.preproc_run_name = preproc_run_name
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.stride = stride

    def setup(self, stage=None):
        split_dict = read_json(S3_BUCKET_NAME, f"{S3_DATA_DIR}/{self.preproc_run_name}/{SPLITS_FILENAME}")
        
        self.train_dataset = VideoDataset(split_dict["train"], self.seq_length, self.stride)
        self.val_dataset = VideoDataset(split_dict["validation"], self.seq_length, self.stride)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
