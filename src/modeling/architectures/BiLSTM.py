import torch.nn as nn

class BiLSTMSegmentation(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits