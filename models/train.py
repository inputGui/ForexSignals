import torch
import torch.nn as nn
import torch.optim as optim
from models.lstm import LSTMModel

def train_model(train_loader, val_loader, input_dim, hidden_dim, layer_dim, output_dim):
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop here
    
    torch.save(model.state_dict(), 'forex_model.pth')
    return model