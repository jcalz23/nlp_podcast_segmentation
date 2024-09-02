import os
import json
import sys
import torch
import torch.nn as nn
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.dataloader import VideoDataModule
from utils.metrics import windiff
from modeling.architectures.BiLSTM import BiLSTMSegmentation
from modeling.architectures.Transformer import TransformerSegmentation
from constants import *
from utils.aws import save_json_to_s3, save_local_to_s3


class PodcastSegmentationModel(pl.LightningModule):
    """
    A PyTorch Lightning module for podcast segmentation.

    This model uses a bidirectional LSTM to process sentence embeddings
    and predict segment/topic transitions in podcasts.
    """

    def __init__(self, model_type, input_dim, hidden_dim, num_layers, num_heads=8, dropout=0.1, window_size=5, segment_threshold=0.5, pos_weight=10.0, learning_rate=1e-3, count_loss_weight=0.1):
        """
        Initialize the PodcastSegmentationModel.

        Args:
            model_type (str): Type of model to use ('bilstm' or 'transformer').
            input_dim (int): Dimension of input features (sentence embeddings).
            hidden_dim (int): Dimension of LSTM hidden state.
            num_layers (int): Number of LSTM layers.
            num_heads (int): Number of attention heads for Transformer model.
            dropout (float): Dropout rate for LSTM layers.
            window_size (int): Window size for WinDiff metric calculation.
            segment_threshold (float): Threshold for binarizing predictions.
            pos_weight (float): Weight for positive class in loss function.
            learning_rate (float): Learning rate for the Adam optimizer.
            count_loss_weight (float): Weight for the count loss term in the loss function.
        """
        super().__init__()
        self.model_type = model_type
        if model_type == 'bilstm':
            self.model = BiLSTMSegmentation(input_dim, hidden_dim, num_layers, dropout)
        elif model_type == 'transformer':
            self.model = TransformerSegmentation(input_dim, hidden_dim, num_layers, num_heads, dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.window_size = window_size
        self.segment_threshold = segment_threshold
        self.pos_weight = pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
        self.learning_rate = learning_rate
        self.validation_step_outputs = []
        self.count_loss_weight = count_loss_weight

    def _compute_loss(self, logits, targets):
        """
        Compute the total loss, which is a combination of the BCE loss and the count loss.

        Args:
            logits (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth segment indicators.

        Returns:
            torch.Tensor: Total loss.
        """
        bce_loss = self.loss_fn(logits, targets)
        pred_count = torch.sigmoid(logits).gt(self.segment_threshold).float().sum()
        true_count = targets.sum()
        count_loss = torch.abs(pred_count - true_count) / targets.numel()
        total_loss = bce_loss + self.count_loss_weight * count_loss
        return total_loss

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, 1).
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (dict): A dictionary containing 'sentence_embeddings' and 'segment_indicators'.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for this step.
        """
        sentence_embeddings = batch['sentence_embeddings']
        segment_indicators = batch['segment_indicators'].float().unsqueeze(-1)
        logits = self(sentence_embeddings)
        
        loss = self._compute_loss(logits, segment_indicators)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (dict): A dictionary containing 'sentence_embeddings' and 'segment_indicators'.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: A dictionary containing the loss and predictions for this step.
        """
        sentence_embeddings = batch['sentence_embeddings']
        segment_indicators = batch['segment_indicators'].float().unsqueeze(-1)
        logits = self(sentence_embeddings)
        
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
        """
        Compute and log metrics at the end of each validation epoch.
        """
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
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The Adam optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class PodcastSegmentationTrainer:
    """
    A trainer class for the PodcastSegmentationModel.

    This class handles the setup and execution of the training process.
    """

    def __init__(self, config=None):
        """
        Initialize the PodcastSegmentationTrainer.

        Args:
            preproc_run_name (str): Name of the preprocessing run.
            config (dict, optional): Configuration dictionary. If None, DEFAULT_CONFIG is used.
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = VideoDataModule(
            self.config['train_data_folder'],
            self.config['batch_size'],
            self.config['seq_length'],
            self.config['stride']
        )
        self.model = PodcastSegmentationModel(
            model_type=self.config['model_type'],
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout'],
            window_size=self.config['window_size'],
            segment_threshold=self.config['segment_threshold'],
            pos_weight=self.config['pos_weight'],
            learning_rate=self.config['learning_rate'],
            count_loss_weight=self.config['count_loss_weight']
        )
        self.max_epochs = self.config['max_epochs']

    def train(self):
        """
        Execute the training process.

        This method sets up callbacks, loggers, and the PyTorch Lightning Trainer,
        then starts the model training.
        """
        # Set up model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='podcast_segmentation-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )

        # Set up early stopping
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )

        # Set up TensorBoard logging
        logger = TensorBoardLogger("lightning_logs", name="podcast_segmentation")

        # Determine the accelerator and devices to use
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = None

        # Initialize variables to track best metrics
        best_val_loss = float('inf')
        best_windiff = float('inf')
        best_epoch = -1

        # Custom callback to track best metrics
        class BestMetricsCallback(pl.Callback):
            def on_validation_end(self, trainer, pl_module):
                nonlocal best_val_loss, best_windiff, best_epoch
                current_val_loss = trainer.callback_metrics['val_loss'].item()
                current_windiff = trainer.callback_metrics['val_windiff'].item()
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_windiff = current_windiff
                    best_epoch = trainer.current_epoch

        # Add the custom callback to the list of callbacks
        callbacks = [checkpoint_callback, early_stop_callback, BestMetricsCallback()]

        # Initialize the PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=True,  # Enable progress bar
            log_every_n_steps=10  # Log every 10 steps
        )

        # Start the training process
        trainer.fit(self.model, self.data_module)

        # Print best model's performance and checkpoint path
        print(f"Best model's performance - Val Loss: {best_val_loss:.4f}, "
              f"WinDiff: {best_windiff:.4f}, Epoch: {best_epoch}")
        print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")

        # Save the config and checkpoint to S3
        config_name = os.path.splitext(os.path.basename(args.config_path))[0]
        s3_config_key = f"{S3_MODELS_DIR}/{config_name}/config.json"
        save_json_to_s3(config, S3_BUCKET_NAME, s3_config_key)
        print(f"Config saved to S3: {s3_config_key}")

        # Save best model checkpoint and metrics to S3
        if checkpoint_callback.best_model_path:
            # Save checkpoint
            s3_checkpoint_key = f"{S3_MODELS_DIR}/{config_name}/best_model.ckpt"
            save_local_to_s3(checkpoint_callback.best_model_path, S3_BUCKET_NAME, s3_checkpoint_key)
            print(f"Best model checkpoint saved to S3: {s3_checkpoint_key}")

            # Save metrics
            metrics = {
                'val_loss': best_val_loss,
                'val_windiff': best_windiff,
                'epoch': best_epoch
            }
            s3_metrics_key = f"{S3_MODELS_DIR}/{config_name}/best_model_metrics.json"
            save_json_to_s3(metrics, S3_BUCKET_NAME, s3_metrics_key)
            print(f"Best model metrics saved to S3: {s3_metrics_key}")
        else:
            print("No best model checkpoint found to save.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train the Podcast Segmentation Model')
    parser.add_argument('--config_path', type=str, help='Path to the configuration JSON file')
    args = parser.parse_args()

    # Load the config file
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Train
    trainer = PodcastSegmentationTrainer(config=config)
    trainer.train()
