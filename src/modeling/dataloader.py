import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class VideoDataset(Dataset):
    def __init__(self, video_ids, data_dir, seq_length=256, stride=128):
        self.video_ids = video_ids
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.stride = stride
        self.data = self._load_all_data()

    def _load_all_data(self):
        all_data = []
        for video_id in self.video_ids:
            with open(f"{self.data_dir}/podcasts/{video_id}.json", "r") as f:
                data = json.load(f)
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
                else:
                    segment = sentence_embedding[start:end]
                    segment_ind = segment_inds[start:end]
                
                all_data.append({
                    "sentence_embeddings": segment,
                    "segment_indicators": segment_ind,
                    "video_id": video_id
                })
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, seq_length=256, stride=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.stride = stride

    def setup(self, stage=None):
        with open(f"{self.data_dir}/split_dict.json", "r") as f:
            split_dict = json.load(f)
        
        self.train_dataset = VideoDataset(split_dict["train"], self.data_dir, self.seq_length, self.stride)
        self.val_dataset = VideoDataset(split_dict["validation"], self.data_dir, self.seq_length, self.stride)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Usage example:
# data_module = VideoDataModule(data_dir="path/to/data", batch_size=64, seq_length=256, stride=128)
