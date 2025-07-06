# src/model.py

import torch
import torch.nn as nn

class EmotionBiLSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, num_classes=8, dropout=0.3):
        super(EmotionBiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # bidirectional → hidden_dim * 2
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # x: (batch, seq_len, input_dim)
        last_time_step = lstm_out[:, -1, :]  # take output from last time step
        out = self.fc(last_time_step)
        return self.softmax(out)
