import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import VideoDataModule
import numpy as np
from scipy.ndimage import convolve1d

def windiff(y_true, y_pred, window_size):
    """
    Calculate the WinDiff metric for segmentation evaluation.

    Args:
        y_true (np.array): Ground truth segmentation.
        y_pred (np.array): Predicted segmentation.
        window_size (int): Size of the comparison window.

    Returns:
        float: WinDiff score (lower is better).
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    window = np.ones(window_size)
    true_counts = convolve1d(y_true, window, mode='constant', cval=0)
    pred_counts = convolve1d(y_pred, window, mode='constant', cval=0)
    
    differences = np.abs(true_counts - pred_counts) > 0
    return np.sum(differences) / (len(y_true) - window_size + 1)

class PodcastSegmentationModel(pl.LightningModule):
    """
    A PyTorch Lightning module for podcast segmentation.

    This model uses a bidirectional LSTM to process sentence embeddings
    and predict segment/topic transitions in podcasts.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        """
        Initialize the PodcastSegmentationModel.

        Args:
            input_dim (int): Dimension of input features (sentence embeddings).
            hidden_dim (int): Dimension of LSTM hidden state.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate for LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()
        self.validation_step_outputs = []  # Add this line to store outputs

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, 1).
        """
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return self.sigmoid(logits)

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
        predictions = self(sentence_embeddings)
        loss = nn.BCELoss()(predictions, segment_indicators)
        self.log('train_loss', loss)
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
        predictions = self(sentence_embeddings)
        loss = nn.BCELoss()(predictions, segment_indicators)
        self.log('val_loss', loss)
        # Store the outputs
        self.validation_step_outputs.append({'val_loss': loss, 'preds': predictions, 'targets': segment_indicators})
        return loss

    def on_validation_epoch_end(self):
        """
        Compute the WinDiff metric at the end of the validation epoch.

        Args:
            outputs (list): A list of dictionaries containing the outputs of validation_step.
        """
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).cpu().numpy()
        
        # Flatten and binarize predictions
        all_preds = (all_preds.flatten() > 0.5).astype(int)
        all_targets = all_targets.flatten().astype(int)
        
        # Compute WinDiff
        window_size = 10  # You may want to adjust this based on your specific use case
        windiff_score = windiff(all_targets, all_preds, window_size)
        
        self.log('val_windiff', windiff_score)
        
        # Compute average validation loss
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        
        # Print metrics
        print(f"Epoch {self.current_epoch} - Avg Val Loss: {avg_loss:.4f}, WinDiff: {windiff_score:.4f}")
        
        # Clear the outputs list
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The Adam optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class PodcastSegmentationTrainer:
    """
    A trainer class for the PodcastSegmentationModel.

    This class handles the setup and execution of the training process.
    """

    def __init__(self, data_dir, batch_size=32, seq_length=256, stride=128, max_epochs=10):
        """
        Initialize the PodcastSegmentationTrainer.

        Args:
            data_dir (str): Directory containing the dataset.
            batch_size (int): Batch size for training and validation.
            seq_length (int): Sequence length for input data.
            stride (int): Stride for sliding window in data preparation.
            max_epochs (int): Maximum number of training epochs.
        """
        self.data_module = VideoDataModule(data_dir, batch_size, seq_length, stride)
        self.model = PodcastSegmentationModel(input_dim=384, hidden_dim=256, num_layers=2)  # Assuming BERT embeddings (768-dim)
        self.max_epochs = max_epochs

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

        # Initialize the PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=True,  # Enable progress bar
            log_every_n_steps=10  # Log every 10 steps
        )

        # Start the training process
        trainer.fit(self.model, self.data_module)

        # Print best model's performance
        print(f"Best model's performance - Val Loss: {trainer.callback_metrics['val_loss']:.4f}, "
              f"WinDiff: {trainer.callback_metrics['val_windiff']:.4f}")

if __name__ == "__main__":
    # Example usage
    trainer = PodcastSegmentationTrainer(
        data_dir="../preprocessing", batch_size=64, seq_length=256, stride=128, max_epochs=20
        )
    trainer.train()
