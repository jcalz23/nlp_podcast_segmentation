import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.aws import read_json
from constants import S3_BUCKET_NAME, S3_DATA_DIR, SPLITS_FILENAME
from transformers import BertTokenizer, BertModel

class LSTMDataset(Dataset):
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
                else:
                    segment = sentence_embedding[start:end]
                    segment_ind = segment_inds[start:end]

                # Force the first sentence to be a segment start
                segment_ind[0] = 1
                
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

class BERTDataset(Dataset):
    def __init__(self, video_ids, tokenizer, max_length=128, seq_length=512, stride=256):
        self.video_ids = video_ids
        self.tokenizer = tokenizer
        self.max_length = max_length  # Max length for each sentence
        self.seq_length = seq_length  # Number of sentences in a sequence
        self.stride = stride
        self.data = self._load_all_data()
        logging.info(f"Finished loading {len(self.data)} segments from {len(video_ids)} videos")

    def _load_all_data(self):
        all_data = []
        for video_id in self.video_ids:
            # Load inputs, outputs
            data = read_json(S3_BUCKET_NAME, f"{S3_DATA_DIR}/podcasts/{video_id}.json")
            sentences = data["sentences"][:self.seq_length]
            segment_inds = torch.tensor(data["segment_indicators"][:self.seq_length])
            
            # Tokenize each sentence
            encoded_inputs = self.tokenizer(
                sentences, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
            )

            # Pad sequences if sequence length if too short
            if len(segment_inds) < self.seq_length:
                pass
                # padding_length = self.seq_length - len(segment_inds) - 1
                # encoded_inputs['input_ids'] = torch.cat([
                #     encoded_inputs['input_ids'],
                #     torch.zeros(padding_length, self.max_length, dtype=torch.long)
                # ])
                # encoded_inputs['attention_mask'] = torch.cat([
                #     encoded_inputs['attention_mask'],
                #     torch.zeros(padding_length, self.max_length, dtype=torch.long)
                # ])
            else:
                # Add to batch
                all_data.append({
                    "input_ids": encoded_inputs['input_ids'],
                    "attention_mask": encoded_inputs['attention_mask'],
                    "segment_indicators": segment_inds,
                    "video_id": video_id
                })

            
            # # Create multiple records for long podcasts
            # for start in range(0, len(segment_inds), self.stride):
            #     # Get end index
            #     end = start + self.seq_length

            #     # Encoded sentences
            #     encoded_inputs = self.tokenizer(
            #         sentences[start:end], padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
            #     )

            #     # Pad the last segment if it's shorter than seq_length
            #     if end > len(segment_inds):
            #         padding_length = end - len(segment_inds)
            #         segment_input_ids = torch.cat([
            #             encoded_inputs['input_ids'],
            #             torch.zeros(padding_length, self.max_length, dtype=torch.long)
            #         ]).view(-1, self.max_length)
            #         segment_attention_mask = torch.cat([
            #             encoded_inputs['attention_mask'],
            #             torch.zeros(padding_length, self.max_length, dtype=torch.long)
            #         ]).view(-1, self.max_length)
            #         segment_ind = torch.cat([
            #             segment_inds[start:end],
            #             torch.zeros(padding_length)
            #         ])
            #     else:
            #         segment_input_ids = encoded_inputs['input_ids']
            #         segment_attention_mask = encoded_inputs['attention_mask']
            #         segment_ind = segment_inds[start:end]

            #     # Force the first sentence to be a segment start
            #     segment_ind[0] = 1

            #     # If segment is < 512 print
            #     if len(segment_ind) < self.seq_length:
            #         print(f"\n\n\nsegment_input_ids: {segment_input_ids.shape}\n\n\n")
            #         print(f"segment_attention_mask: {segment_attention_mask.shape}\n\n\n")
            #         print(f"segment_ind: {segment_ind.shape}\n\n\n")

            #     all_data.append({
            #         "input_ids": segment_input_ids,
            #         "attention_mask": segment_attention_mask,
            #         "segment_indicators": segment_ind,
            #         "video_id": video_id
            #     })
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, preproc_run_name, batch_size=32, seq_length=512, stride=256, model_type='lstm'):
        super().__init__()
        self.preproc_run_name = preproc_run_name
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.stride = stride
        self.model_type = model_type
        if model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def setup(self, stage=None):
        split_dict = read_json(S3_BUCKET_NAME, f"{S3_DATA_DIR}/{self.preproc_run_name}/{SPLITS_FILENAME}")
        
        if self.model_type == 'lstm':
            self.train_dataset = LSTMDataset(split_dict["train"], self.seq_length, self.stride)
            self.val_dataset = LSTMDataset(split_dict["validation"], self.seq_length, self.stride)
        elif self.model_type == 'bert':
            self.train_dataset = BERTDataset(split_dict["train"], self.tokenizer, self.seq_length)
            self.val_dataset = BERTDataset(split_dict["validation"], self.tokenizer, self.seq_length)
        else:
            raise ValueError("model_type must be either 'lstm' or 'bert'")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
