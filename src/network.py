"""
    This file contains a neural network module for us to
    define our actor and critic networks in PPO.
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



class LinearWarmupCosineDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = optimizer.defaults['lr']
        super(LinearWarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr = (self.max_lr - self.min_lr) * self.last_epoch / self.warmup_steps + self.min_lr
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return [lr for _ in self.optimizer.param_groups]


class FeedForwardNN(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, obs_dim, act_dim):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(obs_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, act_dim)

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input

            Return:
                output - the output of our forward pass
        """
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output
    

class TransformerNN(nn.Module):
    """
    A neural network that uses a Transformer encoder for processing sequences.
    This network includes positional encodings to maintain sequence order information.
    """
    def __init__(self, act_dim=1, emb_dim=128, obs_dim=400, nhead=4, num_encoder_layers=4, max_seq_length=5000):
    # def __init__(self, max_seq_length=5000):
        """
        Initialize the network and set up the layers.

        Parameters:
            act_dim - input dimensions as an int
            out_dim - output dimensions as an int
            seq_length - the length of the input sequences
            nhead - the number of heads in the multiheadattention models
            num_encoder_layers - the number of sub-encoder-layers in the encoder
            max_seq_length - maximum length of the input sequences for positional encoding
        """
        super(TransformerNN, self).__init__()

        self.seq_length = obs_dim
        self.embedding = nn.Linear(1, emb_dim)  # Embedding layer

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Positional Encoding
        self.positional_encoding = self._generate_positional_encoding(emb_dim, max_seq_length)

        # Decoder layer to get the final output
        self.decoder = nn.Linear(emb_dim * obs_dim, act_dim)


    def _generate_positional_encoding(self, dim, max_len):
        """
        Generate and return positional encoding matrix.
        Shape: (max_len, dim)
        """
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, dim)
        return pe

    def forward(self, obs):
        """
        Runs a forward pass on the neural network.

        Parameters:
            obs - observation to pass as input (expected shape: [batch_size, seq_length, act_dim])

        Return:
            output - the output of our forward pass
        """
        # Embedding the input
        # obs_shape = (seq_length)
        obs_shape = obs.shape
        if len(obs_shape) == 1:
            obs = obs.unsqueeze(0).unsqueeze(-1)
        if len(obs_shape) == 2:
            obs = obs.unsqueeze(-1)
        
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print("Warning: obs contains NaNs or infinities")
        # obs_shape = (batch_size, seq_length, 1)
        try:
            embedded = self.embedding(obs)  # Shape: [batch_size, seq_length, act_dim]
        except:
            print(f"Error with obs: {obs}")
            exit()

        # Transformer requires input shape [seq_length, batch_size, act_dim]
        embedded = embedded.permute(1, 0, 2) + self.positional_encoding[:self.seq_length, :].to(obs.device)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(embedded)

        # Reshape and decode the transformer output
        transformer_output = transformer_output.permute(1, 0, 2).contiguous()
        transformer_output = transformer_output.view(transformer_output.size(0), -1)
        output = self.decoder(transformer_output)
        if len(obs_shape) == 1:
            output = output.squeeze(0)

        return output


if __name__ == '__main__':
    # Test the network
    obs_dim = 400
    act_dim = 56
    obs = torch.randn(obs_dim)
    obs = obs.unsqueeze(0)
    obs = obs.unsqueeze(-1)
    # policy = FeedForwardNN(obs_dim, act_dim)
    policy = TransformerNN(act_dim=act_dim, 
                           emb_dim=256,
                           obs_dim=400, 
                           nhead=8, 
                           num_encoder_layers=4, 
                           max_seq_length=5000)
    output = policy(obs)
    print(f"Output shape: {output.shape}")