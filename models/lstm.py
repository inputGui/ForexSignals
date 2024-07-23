import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction.

    Attributes:
        hidden_dim (int): The number of features in the hidden state h.
        layer_dim (int): The number of LSTM layers.
        lstm (nn.LSTM): LSTM layers.
        fc (nn.Linear): Fully connected layer for output.
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        Initialize the LSTM model.

        Args:
            input_dim (int): The number of input features.
            hidden_dim (int): The number of features in the hidden state h.
            layer_dim (int): The number of LSTM layers.
            output_dim (int): The number of output features.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out