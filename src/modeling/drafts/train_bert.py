import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.dataloader import VideoDataModule
from utils.metrics import windiff
from modeling.architectures.BiLSTM import BiLSTMSegmentation
from transformers import BertModel, BertTokenizer

class BertBasedSegmentation(nn.Module):
    """A BERT-based model for text segmentation."""

    def __init__(self, model_name='bert-base-uncased', hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state

        # Pass through LSTM
        lstm_output, _ = self.lstm(sequence_output)

        # Predict binary segmentation labels
        logits = self.classifier(lstm_output)
        return logits

class PodcastSegmentationModel(pl.LightningModule):
    def __init__(self, model_type='lstm', input_dim=384, hidden_dim=256, num_layers=4, dropout=0.1, window_size=5, segment_threshold=0.5, pos_weight=10.0, learning_rate=1e-3, count_loss_weight=0.1):
        super().__init__()
        self.model_type = model_type
        if model_type == 'lstm':
            self.model = BiLSTMSegmentation(input_dim, hidden_dim, num_layers, dropout)
        elif model_type == 'bert':
            self.model = BertBasedSegmentation()
        else:
            raise ValueError("model_type must be either 'lstm' or 'bert'")
        
        self.window_size = window_size
        self.segment_threshold = segment_threshold
        self.pos_weight = pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
        self.learning_rate = learning_rate
        self.validation_step_outputs = []
        self.count_loss_weight = count_loss_weight

    def _compute_loss(self, logits, targets):
        bce_loss = self.loss_fn(logits, targets)
        pred_count = torch.sigmoid(logits).gt(self.segment_threshold).float().sum()
        true_count = targets.sum()
        count_loss = torch.abs(pred_count - true_count) / targets.numel()
        total_loss = bce_loss + self.count_loss_weight * count_loss
        return total_loss

    def forward(self, x):
        if self.model_type == 'lstm':
            return self.model(x)
        elif self.model_type == 'bert':
            return self.model(x['input_ids'], x['attention_mask'])

    def training_step(self, batch, batch_idx):
        if self.model_type == 'lstm':
            sentence_embeddings = batch['sentence_embeddings']
            print(f"Sentence Embeddings shape: {sentence_embeddings.shape}")
            segment_indicators = batch['segment_indicators'].float().unsqueeze(-1)
            logits = self(sentence_embeddings)
        elif self.model_type == 'bert':
            input_ids = batch['input_ids'].squeeze(0)
            attention_mask = batch['attention_mask'].squeeze(0)
            segment_indicators = batch['segment_indicators'].float().unsqueeze(-1)
            print(f"\n\n\nInput IDs shape: {input_ids.shape}")
            print(f"Attention Mask shape: {attention_mask.shape}")
            print(f"Segment Indicators shape: {segment_indicators.shape}")
            logits = self({'input_ids': input_ids, 'attention_mask': attention_mask})
        
        loss = self._compute_loss(logits, segment_indicators)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_type == 'lstm':
            sentence_embeddings = batch['sentence_embeddings']
            segment_indicators = batch['segment_indicators'].float().unsqueeze(-1)
            logits = self(sentence_embeddings)
        elif self.model_type == 'bert':
            input_ids = batch['input_ids'].squeeze(0)
            attention_mask = batch['attention_mask'].squeeze(0)
            segment_indicators = batch['segment_indicators'].float().unsqueeze(-1)
            logits = self({'input_ids': input_ids, 'attention_mask': attention_mask})
        
        loss = self._compute_loss(logits, segment_indicators)
        predictions = torch.sigmoid(logits)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.validation_step_outputs.append({
            'val_loss': loss,
            'preds': predictions,
            'targets': segment_indicators
        })
        
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).cpu().numpy()
        
        all_preds = (all_preds.flatten() > self.segment_threshold).astype(int)
        all_targets = all_targets.flatten().astype(int)
        
        windiff_score = windiff(all_targets, all_preds, self.window_size)
        self.log('val_windiff', windiff_score, prog_bar=True, logger=True)
        
        avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.log('avg_val_loss', avg_val_loss)
        
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', float('nan'))
        print(f"Epoch {self.current_epoch} - "
              f"Avg Train Loss: {train_loss:.4f}, "
              f"Avg Val Loss: {avg_val_loss:.4f}, "
              f"WinDiff: {windiff_score:.4f}")
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class PodcastSegmentationTrainer:
    def __init__(self, preproc_run_name, model_type='lstm', batch_size=32, seq_length=256, stride=128, max_epochs=10, window_size=5, segment_threshold=0.5, pos_weight=10.0, learning_rate=1e-3, count_loss_weight=0.1):
        self.data_module = VideoDataModule(preproc_run_name, batch_size, seq_length, stride, model_type=model_type)
        self.model = PodcastSegmentationModel(
            model_type=model_type,
            input_dim=384 if model_type == 'lstm' else None,
            hidden_dim=256, num_layers=4,
            window_size=window_size, segment_threshold=segment_threshold,
            pos_weight=pos_weight, learning_rate=learning_rate,
            count_loss_weight=count_loss_weight
        )
        self.max_epochs = max_epochs

    def train(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='podcast_segmentation-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )

        logger = TensorBoardLogger("lightning_logs", name="podcast_segmentation")

        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = None

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=True,
            log_every_n_steps=10
        )

        trainer.fit(self.model, self.data_module)

        print(f"Best model's performance - Val Loss: {trainer.callback_metrics['val_loss']:.4f}, "
              f"WinDiff: {trainer.callback_metrics['val_windiff']:.4f}")
        print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    preproc_run_name = input("Enter the preprocessing run name: ")
    model_type = input("Enter model type (lstm or bert): ")
    trainer = PodcastSegmentationTrainer(
        preproc_run_name=preproc_run_name,
        model_type=model_type,
        batch_size=64 if model_type == 'lstm' else 16,
        seq_length=256, stride=128, max_epochs=100,
        window_size=5, segment_threshold=0.5, pos_weight=20.0,
        learning_rate=1e-3 if model_type == 'lstm' else 2e-5,
        count_loss_weight=0.2
    )
    trainer.train()
