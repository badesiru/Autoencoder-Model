#feedforward autoencoder using pyrotch, rectructs benign network traffic
#siri


import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super(Autoencoder, self).__init__()

        #Sanity check for hyperparameters
        print("This is hidden DIMS:", hidden_dims)
        print("\nThis is input DIM:", input_dim)
        print("\nThis is latent DIM:", latent_dim)

        encoder_layers = [] 
        prev_dim = input_dim

        #Loop to backpropogate through hidden layers
        for h_dim in hidden_dims: 
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim)) #end at the latent dim
        self.encoder = nn.Sequential(*encoder_layers)

        #Decoder layering to reconstruct
        decoder_layers = []
        prev_dim = latent_dim

        #Loop to backpropogate through hidden layers in reverse, starting at latent dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim)) #end at original input dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)