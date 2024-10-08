import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, (hn, cn) = self.lstm(x)  # LSTM returns output, (hidden state, cell state)
        out = self.fc(out[:, -1, :])  # Use the last time step output for prediction
        return out